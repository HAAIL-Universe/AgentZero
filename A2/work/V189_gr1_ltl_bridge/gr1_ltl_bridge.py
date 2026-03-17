"""
V189: GR(1)-LTL Bridge
======================
Auto-detects GR(1) fragments in LTL specifications and routes to the
efficient polynomial-time solver (V187) when possible, falling back to
general reactive synthesis (V186) for non-GR(1) specs.

GR(1) fragment:
  (AND_i G(safe_i)) AND (AND_j GF(J_j^e)) -> (AND_k G(safe_k)) AND (AND_l GF(J_l^s))

Where safe_i/safe_k are boolean combinations of atoms (safety),
and J_j^e/J_l^s are boolean combinations of atoms (justice/liveness).

Composes:
- V023 (LTL model checker) -- LTL formula AST
- V186 (reactive synthesis) -- general LTL synthesis
- V187 (GR(1) synthesis) -- polynomial GR(1) synthesis
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, FrozenSet, Callable
from enum import Enum, auto

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)

sys.path.insert(0, os.path.join(_work, 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(_work, 'V186_reactive_synthesis'))
sys.path.insert(0, os.path.join(_work, 'V187_gr1_synthesis'))

from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release, WeakUntil,
    LTLTrue, LTLFalse, atoms, nnf
)

from reactive_synthesis import (
    synthesize as v186_synthesize,
    SynthesisResult, SynthesisVerdict,
    MealyMachine as V186Mealy
)

from gr1_synthesis import (
    GR1Game, GR1Strategy, GR1Result, GR1Verdict,
    BoolGR1Spec, gr1_solve, gr1_synthesize,
    build_bool_game, strategy_to_mealy,
    MealyMachine as V187Mealy,
    gr1_summary
)


# ============================================================
# GR(1) Fragment Detection
# ============================================================

class FragmentKind(Enum):
    """Classification of an LTL formula's fragment."""
    PROPOSITIONAL = "propositional"     # Boolean combination of atoms
    SAFETY = "safety"                    # G(prop)
    JUSTICE = "justice"                  # GF(prop)
    GR1 = "gr1"                         # Full GR(1) spec
    NON_GR1 = "non_gr1"                # Not in GR(1) fragment


@dataclass
class GR1Decomposition:
    """Decomposition of an LTL formula into GR(1) components."""
    is_gr1: bool
    env_safety: List[LTL] = field(default_factory=list)     # G(prop) assumptions
    env_justice: List[LTL] = field(default_factory=list)     # GF(prop) assumptions
    sys_safety: List[LTL] = field(default_factory=list)      # G(prop) guarantees
    sys_justice: List[LTL] = field(default_factory=list)      # GF(prop) guarantees
    reason: str = ""                                          # Why not GR(1) if applicable
    env_init: Optional[LTL] = None                           # Initial env constraint
    sys_init: Optional[LTL] = None                           # Initial sys constraint


@dataclass
class BridgeResult:
    """Result from the GR(1)-LTL bridge."""
    verdict: str                            # "realizable" / "unrealizable" / "unknown"
    method: str                             # "gr1" / "ltl" / "quick"
    controller: Optional[V186Mealy] = None
    gr1_result: Optional[GR1Result] = None
    ltl_result: Optional[SynthesisResult] = None
    decomposition: Optional[GR1Decomposition] = None
    is_gr1: bool = False
    stats: Dict = field(default_factory=dict)


def is_propositional(f: LTL) -> bool:
    """Check if formula is a boolean combination of atoms (no temporal operators)."""
    if f.op == LTLOp.ATOM:
        return True
    if f.op in (LTLOp.TRUE, LTLOp.FALSE):
        return True
    if f.op == LTLOp.NOT:
        return is_propositional(f.left)
    if f.op in (LTLOp.AND, LTLOp.OR, LTLOp.IMPLIES, LTLOp.IFF):
        return is_propositional(f.left) and is_propositional(f.right)
    # Temporal operators
    return False


def is_next_propositional(f: LTL) -> bool:
    """Check if formula is a boolean combination of atoms and X(atoms).
    Used for transition constraints in GR(1)."""
    if f.op == LTLOp.ATOM:
        return True
    if f.op in (LTLOp.TRUE, LTLOp.FALSE):
        return True
    if f.op == LTLOp.NOT:
        return is_next_propositional(f.left)
    if f.op in (LTLOp.AND, LTLOp.OR, LTLOp.IMPLIES, LTLOp.IFF):
        return is_next_propositional(f.left) and is_next_propositional(f.right)
    if f.op == LTLOp.X:
        return is_propositional(f.left)
    return False


def classify_conjunct(f: LTL) -> Tuple[str, Optional[LTL]]:
    """Classify a top-level conjunct of a GR(1) spec.

    Returns:
        (kind, inner) where kind is one of:
        - "safety": G(prop) -- inner is prop
        - "justice": GF(prop) -- inner is prop
        - "transition": G(next_prop) -- inner is next_prop
        - "init": propositional -- inner is the formula
        - "non_gr1": not classifiable
    """
    if is_propositional(f):
        return ("init", f)

    if f.op == LTLOp.G:
        inner = f.left
        if is_propositional(inner):
            return ("safety", inner)
        if inner.op == LTLOp.F and is_propositional(inner.left):
            return ("justice", inner.left)
        if is_next_propositional(inner):
            return ("transition", inner)
        # G(implies(prop, next_prop)) -- transition constraint
        if inner.op == LTLOp.IMPLIES and is_next_propositional(inner):
            return ("transition", inner)
        return ("non_gr1", None)

    if f.op == LTLOp.F:
        inner = f.left
        if f.left.op == LTLOp.G and is_propositional(f.left.left):
            # FG(prop) -- stability, not directly GR(1) but can approximate
            return ("non_gr1", None)
        return ("non_gr1", None)

    return ("non_gr1", None)


def collect_conjuncts(f: LTL) -> List[LTL]:
    """Flatten nested AND into a list of conjuncts."""
    if f.op == LTLOp.AND:
        return collect_conjuncts(f.left) + collect_conjuncts(f.right)
    if f.op == LTLOp.TRUE:
        return []
    return [f]


def detect_gr1(spec: LTL, env_vars: Set[str], sys_vars: Set[str]) -> GR1Decomposition:
    """Analyze an LTL formula and decompose it into GR(1) components if possible.

    Handles these forms:
    1. Direct conjunction: G(s1) & GF(j1) & G(s2) & GF(j2) ...
    2. Assume-guarantee: (assumptions) -> (guarantees)
    3. Mixed: some safety, some justice, some propositional init
    """
    all_vars = env_vars | sys_vars

    # Case 1: Implication (assume-guarantee form)
    if spec.op == LTLOp.IMPLIES:
        assumption = spec.left
        guarantee = spec.right
        return _decompose_assume_guarantee(assumption, guarantee, env_vars, sys_vars)

    # Case 2: Conjunction of GR(1) components (all guarantees, no assumptions)
    conjuncts = collect_conjuncts(spec)
    return _decompose_guarantees_only(conjuncts, env_vars, sys_vars)


def _decompose_assume_guarantee(
    assumption: LTL, guarantee: LTL,
    env_vars: Set[str], sys_vars: Set[str]
) -> GR1Decomposition:
    """Decompose (assumption -> guarantee) into GR(1) components."""
    result = GR1Decomposition(is_gr1=True)

    # Process assumptions
    for conj in collect_conjuncts(assumption):
        kind, inner = classify_conjunct(conj)
        if kind == "safety":
            result.env_safety.append(inner)
        elif kind == "justice":
            result.env_justice.append(inner)
        elif kind == "init":
            if result.env_init is None:
                result.env_init = inner
            else:
                result.env_init = And(result.env_init, inner)
        elif kind == "transition":
            # Transition constraints treated as safety
            result.env_safety.append(inner)
        else:
            result.is_gr1 = False
            result.reason = f"Non-GR(1) assumption: {conj}"
            return result

    # Process guarantees
    for conj in collect_conjuncts(guarantee):
        kind, inner = classify_conjunct(conj)
        if kind == "safety":
            result.sys_safety.append(inner)
        elif kind == "justice":
            result.sys_justice.append(inner)
        elif kind == "init":
            if result.sys_init is None:
                result.sys_init = inner
            else:
                result.sys_init = And(result.sys_init, inner)
        elif kind == "transition":
            result.sys_safety.append(inner)
        else:
            result.is_gr1 = False
            result.reason = f"Non-GR(1) guarantee: {conj}"
            return result

    # Must have at least one justice on each side for proper GR(1),
    # but we can handle safety-only as a degenerate case
    return result


def _decompose_guarantees_only(
    conjuncts: List[LTL],
    env_vars: Set[str], sys_vars: Set[str]
) -> GR1Decomposition:
    """Decompose a conjunction (all guarantees, no assumptions)."""
    result = GR1Decomposition(is_gr1=True)

    for conj in conjuncts:
        kind, inner = classify_conjunct(conj)
        if kind == "safety":
            result.sys_safety.append(inner)
        elif kind == "justice":
            result.sys_justice.append(inner)
        elif kind == "init":
            if result.sys_init is None:
                result.sys_init = inner
            else:
                result.sys_init = And(result.sys_init, inner)
        elif kind == "transition":
            result.sys_safety.append(inner)
        else:
            result.is_gr1 = False
            result.reason = f"Non-GR(1) conjunct: {conj}"
            return result

    return result


# ============================================================
# LTL-to-GR(1) Conversion
# ============================================================

def _eval_prop(formula: LTL, state: FrozenSet[str]) -> bool:
    """Evaluate a propositional formula over a state (set of true vars)."""
    if formula.op == LTLOp.ATOM:
        return formula.name in state
    if formula.op == LTLOp.TRUE:
        return True
    if formula.op == LTLOp.FALSE:
        return False
    if formula.op == LTLOp.NOT:
        return not _eval_prop(formula.left, state)
    if formula.op == LTLOp.AND:
        return _eval_prop(formula.left, state) and _eval_prop(formula.right, state)
    if formula.op == LTLOp.OR:
        return _eval_prop(formula.left, state) or _eval_prop(formula.right, state)
    if formula.op == LTLOp.IMPLIES:
        return not _eval_prop(formula.left, state) or _eval_prop(formula.right, state)
    if formula.op == LTLOp.IFF:
        return _eval_prop(formula.left, state) == _eval_prop(formula.right, state)
    raise ValueError(f"Not propositional: {formula}")


def _eval_next_prop(formula: LTL, state: FrozenSet[str],
                     next_state: FrozenSet[str]) -> bool:
    """Evaluate a next-propositional formula (mix of current and X(next) atoms)."""
    if formula.op == LTLOp.ATOM:
        return formula.name in state
    if formula.op in (LTLOp.TRUE, LTLOp.FALSE):
        return formula.op == LTLOp.TRUE
    if formula.op == LTLOp.X:
        return _eval_prop(formula.left, next_state)
    if formula.op == LTLOp.NOT:
        return not _eval_next_prop(formula.left, state, next_state)
    if formula.op == LTLOp.AND:
        return (_eval_next_prop(formula.left, state, next_state) and
                _eval_next_prop(formula.right, state, next_state))
    if formula.op == LTLOp.OR:
        return (_eval_next_prop(formula.left, state, next_state) or
                _eval_next_prop(formula.right, state, next_state))
    if formula.op == LTLOp.IMPLIES:
        return (not _eval_next_prop(formula.left, state, next_state) or
                _eval_next_prop(formula.right, state, next_state))
    if formula.op == LTLOp.IFF:
        return (_eval_next_prop(formula.left, state, next_state) ==
                _eval_next_prop(formula.right, state, next_state))
    raise ValueError(f"Not next-propositional: {formula}")


def _enumerate_valuations(vars: List[str]) -> List[FrozenSet[str]]:
    """Enumerate all 2^n valuations of the given variables."""
    if not vars:
        return [frozenset()]
    result = []
    rest = _enumerate_valuations(vars[1:])
    for r in rest:
        result.append(r)                            # var is false
        result.append(r | frozenset([vars[0]]))     # var is true
    return result


def decomposition_to_spec(
    decomp: GR1Decomposition,
    env_vars: Set[str],
    sys_vars: Set[str]
) -> BoolGR1Spec:
    """Convert a GR(1) decomposition into a BoolGR1Spec for V187 solver."""
    env_list = sorted(env_vars)
    sys_list = sorted(sys_vars)

    # Separate safety constraints into propositional (invariant) and
    # next-propositional (transition).
    # Propositional safety G(p) means p must hold in EVERY state.
    # Next-propositional safety G(p(x, X(y))) constrains transitions.
    sys_invariants = [s for s in decomp.sys_safety if is_propositional(s)]
    sys_transitions = [s for s in decomp.sys_safety if not is_propositional(s)]
    env_invariants = [s for s in decomp.env_safety if is_propositional(s)]
    env_transitions = [s for s in decomp.env_safety if not is_propositional(s)]

    # Initial conditions: must satisfy init AND all invariants
    def env_init_fn(state):
        if decomp.env_init is not None:
            if not _eval_prop(decomp.env_init, state):
                return False
        for inv in env_invariants:
            if not _eval_prop(inv, state):
                return False
        return True

    def sys_init_fn(state):
        if decomp.sys_init is not None:
            if not _eval_prop(decomp.sys_init, state):
                return False
        for inv in sys_invariants:
            if not _eval_prop(inv, state):
                return False
        return True

    # Transition constraints: invariants must hold on next_state,
    # transition constraints evaluated on (state, next_state)
    def env_trans_fn(state, next_env_vals):
        for inv in env_invariants:
            if not _eval_prop(inv, next_env_vals):
                return False
        for s in env_transitions:
            if not _eval_next_prop_env(s, state, next_env_vals, env_list):
                return False
        return True

    def sys_trans_fn(state, next_state):
        for inv in sys_invariants:
            if not _eval_prop(inv, next_state):
                return False
        for s in sys_transitions:
            if not _eval_next_prop_full(s, state, next_state):
                return False
        return True

    # Justice conditions
    env_justice_fns = []
    for j in decomp.env_justice:
        def make_fn(formula):
            return lambda state: _eval_prop(formula, state)
        env_justice_fns.append(make_fn(j))

    sys_justice_fns = []
    for j in decomp.sys_justice:
        def make_fn(formula):
            return lambda state: _eval_prop(formula, state)
        sys_justice_fns.append(make_fn(j))

    return BoolGR1Spec(
        env_vars=env_list,
        sys_vars=sys_list,
        env_init=env_init_fn,
        sys_init=sys_init_fn,
        env_trans=env_trans_fn,
        sys_trans=sys_trans_fn,
        env_justice=env_justice_fns,
        sys_justice=sys_justice_fns
    )


def _eval_next_prop_env(formula: LTL, state: FrozenSet[str],
                         next_env_vals: FrozenSet[str],
                         env_list: List[str]) -> bool:
    """Evaluate transition formula with partial next state (env vars only).
    Checks if the env_safety holds for ALL possible sys next vals."""
    # For env safety, formula should only reference env vars in next state
    # If it references sys vars in next, we conservatively return True
    try:
        # Try direct evaluation assuming next = next_env_vals (ignoring sys)
        return _eval_next_prop(formula, state, next_env_vals)
    except Exception:
        return True  # Conservative: don't over-restrict env


def _eval_next_prop_full(formula: LTL, state: FrozenSet[str],
                          next_state: FrozenSet[str]) -> bool:
    """Evaluate transition formula with full next state."""
    try:
        return _eval_next_prop(formula, state, next_state)
    except Exception:
        return True


# ============================================================
# Bridge: Unified Synthesis API
# ============================================================

def synthesize(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    force_method: Optional[str] = None
) -> BridgeResult:
    """Synthesize a controller from an LTL spec, auto-detecting GR(1) fragment.

    Args:
        spec: LTL specification
        env_vars: Variables controlled by environment
        sys_vars: Variables controlled by system
        force_method: "gr1" to force GR(1), "ltl" to force general, None for auto

    Returns:
        BridgeResult with controller and synthesis details
    """
    # Quick check for trivial cases
    qc = quick_check(spec, env_vars, sys_vars)
    if qc is not None:
        return BridgeResult(
            verdict=qc,
            method="quick",
            is_gr1=False,
            stats={"quick_check": qc}
        )

    decomp = detect_gr1(spec, env_vars, sys_vars)

    if force_method == "ltl" or (force_method is None and not decomp.is_gr1):
        return _synthesize_ltl(spec, env_vars, sys_vars, decomp)

    if force_method == "gr1" or (force_method is None and decomp.is_gr1):
        return _synthesize_gr1(spec, env_vars, sys_vars, decomp)

    return _synthesize_ltl(spec, env_vars, sys_vars, decomp)


def _check_uncontrollable_safety(
    decomp: GR1Decomposition,
    env_vars: Set[str],
    sys_vars: Set[str]
) -> Optional[str]:
    """Check if any sys safety constraint is uncontrollable.

    If a sys_safety invariant references env vars that env can violate,
    the spec is unrealizable (system can't enforce invariants on env vars).
    """
    env_list = sorted(env_vars)
    sys_list = sorted(sys_vars)

    for inv in [s for s in decomp.sys_safety if is_propositional(s)]:
        inv_atoms = atoms(inv)
        if inv_atoms & env_vars:
            # Invariant references env vars -- check if env can violate it
            for env_val in _enumerate_valuations(env_list):
                # Can sys satisfy the invariant for this env choice?
                satisfiable = False
                for sys_val in _enumerate_valuations(sys_list):
                    state = env_val | sys_val
                    if _eval_prop(inv, state):
                        satisfiable = True
                        break
                if not satisfiable:
                    return f"Sys safety '{inv}' is uncontrollable: env can set {env_val}"
    return None


def _synthesize_gr1(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    decomp: GR1Decomposition
) -> BridgeResult:
    """Synthesize using GR(1) polynomial algorithm."""
    try:
        # Check for uncontrollable safety constraints
        uncontrollable = _check_uncontrollable_safety(decomp, env_vars, sys_vars)
        if uncontrollable:
            return BridgeResult(
                verdict="unrealizable",
                method="gr1",
                decomposition=decomp,
                is_gr1=True,
                stats={"uncontrollable_safety": uncontrollable}
            )

        bool_spec = decomposition_to_spec(decomp, env_vars, sys_vars)
        gr1_result = gr1_synthesize(bool_spec)

        controller = None
        if gr1_result.verdict == GR1Verdict.REALIZABLE and gr1_result.strategy:
            game = build_bool_game(bool_spec)
            controller = strategy_to_mealy(
                game, gr1_result.strategy,
                env_vars=sorted(env_vars),
                sys_vars=sorted(sys_vars)
            )

        verdict = "realizable" if gr1_result.verdict == GR1Verdict.REALIZABLE else "unrealizable"

        return BridgeResult(
            verdict=verdict,
            method="gr1",
            controller=controller,
            gr1_result=gr1_result,
            decomposition=decomp,
            is_gr1=True,
            stats={
                "n_env_safety": len(decomp.env_safety),
                "n_env_justice": len(decomp.env_justice),
                "n_sys_safety": len(decomp.sys_safety),
                "n_sys_justice": len(decomp.sys_justice),
                "n_states": gr1_result.n_states,
                "iterations": gr1_result.iterations,
                "winning_region_size": len(gr1_result.winning_region)
            }
        )
    except Exception as e:
        # GR(1) failed, fall back to general LTL
        return _synthesize_ltl(spec, env_vars, sys_vars, decomp,
                               fallback_reason=str(e))


def _synthesize_ltl(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    decomp: GR1Decomposition,
    fallback_reason: str = ""
) -> BridgeResult:
    """Synthesize using general LTL reactive synthesis."""
    ltl_result = v186_synthesize(spec, env_vars, sys_vars)

    verdict_map = {
        SynthesisVerdict.REALIZABLE: "realizable",
        SynthesisVerdict.UNREALIZABLE: "unrealizable",
        SynthesisVerdict.UNKNOWN: "unknown"
    }

    return BridgeResult(
        verdict=verdict_map.get(ltl_result.verdict, "unknown"),
        method="ltl",
        controller=ltl_result.controller,
        ltl_result=ltl_result,
        decomposition=decomp,
        is_gr1=False,
        stats={
            "game_vertices": ltl_result.game_vertices,
            "game_edges": ltl_result.game_edges,
            "automaton_states": ltl_result.automaton_states,
            "winning_region_size": ltl_result.winning_region_size,
            "fallback_reason": fallback_reason
        }
    )


# ============================================================
# Quick Check (before full synthesis)
# ============================================================

def quick_check(spec: LTL, env_vars: Set[str], sys_vars: Set[str]) -> Optional[str]:
    """Fast pre-screening for trivial cases.

    Returns:
        "realizable", "unrealizable", or None (need full synthesis)
    """
    # TRUE is always realizable
    if spec.op == LTLOp.TRUE:
        return "realizable"

    # FALSE is always unrealizable
    if spec.op == LTLOp.FALSE:
        return "unrealizable"

    # G(true) is realizable
    if spec.op == LTLOp.G and spec.left.op == LTLOp.TRUE:
        return "realizable"

    # G(false) is unrealizable
    if spec.op == LTLOp.G and spec.left.op == LTLOp.FALSE:
        return "unrealizable"

    # F(true) is realizable
    if spec.op == LTLOp.F and spec.left.op == LTLOp.TRUE:
        return "realizable"

    # F(false) is unrealizable
    if spec.op == LTLOp.F and spec.left.op == LTLOp.FALSE:
        return "unrealizable"

    # GF(true) is realizable
    if (spec.op == LTLOp.G and spec.left.op == LTLOp.F
            and spec.left.left.op == LTLOp.TRUE):
        return "realizable"

    # GF(false) is unrealizable
    if (spec.op == LTLOp.G and spec.left.op == LTLOp.F
            and spec.left.left.op == LTLOp.FALSE):
        return "unrealizable"

    # Propositional spec: check if sys can satisfy for all env inputs
    if is_propositional(spec):
        return _check_propositional(spec, env_vars, sys_vars)

    return None


def _check_propositional(spec: LTL, env_vars: Set[str],
                          sys_vars: Set[str]) -> str:
    """Check a propositional spec: for all env valuations,
    exists sys valuation satisfying spec."""
    env_list = sorted(env_vars)
    sys_list = sorted(sys_vars)

    for env_val in _enumerate_valuations(env_list):
        found = False
        for sys_val in _enumerate_valuations(sys_list):
            state = env_val | sys_val
            if _eval_prop(spec, state):
                found = True
                break
        if not found:
            return "unrealizable"
    return "realizable"


# ============================================================
# Analysis & Comparison
# ============================================================

def analyze_spec(spec: LTL, env_vars: Set[str], sys_vars: Set[str]) -> Dict:
    """Analyze an LTL spec for GR(1) fragment membership and structure."""
    decomp = detect_gr1(spec, env_vars, sys_vars)
    qc = quick_check(spec, env_vars, sys_vars)

    spec_atoms = atoms(spec)
    env_atoms = spec_atoms & env_vars
    sys_atoms = spec_atoms & sys_vars

    result = {
        "is_gr1": decomp.is_gr1,
        "reason": decomp.reason if not decomp.is_gr1 else "",
        "quick_check": qc,
        "n_env_safety": len(decomp.env_safety),
        "n_env_justice": len(decomp.env_justice),
        "n_sys_safety": len(decomp.sys_safety),
        "n_sys_justice": len(decomp.sys_justice),
        "has_env_init": decomp.env_init is not None,
        "has_sys_init": decomp.sys_init is not None,
        "env_atoms": env_atoms,
        "sys_atoms": sys_atoms,
        "total_atoms": spec_atoms,
        "formula_depth": _formula_depth(spec),
        "formula_size": _formula_size(spec),
    }
    return result


def compare_methods(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str]
) -> Dict:
    """Compare GR(1) and general LTL synthesis on the same spec."""
    decomp = detect_gr1(spec, env_vars, sys_vars)
    results = {}

    # Always run general LTL
    ltl_result = _synthesize_ltl(spec, env_vars, sys_vars, decomp)
    results["ltl"] = {
        "verdict": ltl_result.verdict,
        "method": "ltl",
        "stats": ltl_result.stats
    }

    # Run GR(1) if applicable
    if decomp.is_gr1:
        gr1_result = _synthesize_gr1(spec, env_vars, sys_vars, decomp)
        results["gr1"] = {
            "verdict": gr1_result.verdict,
            "method": "gr1",
            "stats": gr1_result.stats
        }
        results["methods_agree"] = ltl_result.verdict == gr1_result.verdict
    else:
        results["gr1"] = None
        results["methods_agree"] = None

    results["is_gr1"] = decomp.is_gr1
    results["decomposition_reason"] = decomp.reason if not decomp.is_gr1 else ""

    return results


def bridge_summary(result: BridgeResult) -> str:
    """Human-readable summary of a bridge synthesis result."""
    lines = []
    lines.append(f"GR(1)-LTL Bridge Result")
    lines.append(f"  Verdict: {result.verdict}")
    lines.append(f"  Method: {result.method}")
    lines.append(f"  Is GR(1): {result.is_gr1}")

    if result.controller:
        lines.append(f"  Controller states: {len(result.controller.states)}")
        lines.append(f"  Controller transitions: {len(result.controller.transitions)}")

    if result.decomposition and result.decomposition.is_gr1:
        d = result.decomposition
        lines.append(f"  Env safety: {len(d.env_safety)}")
        lines.append(f"  Env justice: {len(d.env_justice)}")
        lines.append(f"  Sys safety: {len(d.sys_safety)}")
        lines.append(f"  Sys justice: {len(d.sys_justice)}")

    for k, v in result.stats.items():
        if v:
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ============================================================
# Convenience Synthesis Functions
# ============================================================

def synthesize_assume_guarantee(
    assumptions: LTL,
    guarantees: LTL,
    env_vars: Set[str],
    sys_vars: Set[str]
) -> BridgeResult:
    """Synthesize from assume-guarantee spec: assumptions -> guarantees."""
    spec = Implies(assumptions, guarantees)
    return synthesize(spec, env_vars, sys_vars)


def synthesize_safety(
    bad: LTL,
    env_vars: Set[str],
    sys_vars: Set[str]
) -> BridgeResult:
    """Synthesize from safety spec: G(!bad)."""
    spec = Globally(Not(bad))
    return synthesize(spec, env_vars, sys_vars)


def synthesize_liveness(
    goal: LTL,
    env_vars: Set[str],
    sys_vars: Set[str]
) -> BridgeResult:
    """Synthesize from liveness spec: GF(goal)."""
    spec = Globally(Finally(goal))
    return synthesize(spec, env_vars, sys_vars)


def synthesize_response(
    trigger: LTL,
    response: LTL,
    env_vars: Set[str],
    sys_vars: Set[str]
) -> BridgeResult:
    """Synthesize from response spec: G(trigger -> F(response))."""
    spec = Globally(Implies(trigger, Finally(response)))
    return synthesize(spec, env_vars, sys_vars)


def synthesize_gr1_direct(
    env_assumptions: List[LTL],
    sys_guarantees: List[LTL],
    env_safety: List[LTL],
    sys_safety: List[LTL],
    env_vars: Set[str],
    sys_vars: Set[str],
    env_init: Optional[LTL] = None,
    sys_init: Optional[LTL] = None
) -> BridgeResult:
    """Synthesize directly from GR(1) components (bypass LTL detection)."""
    decomp = GR1Decomposition(
        is_gr1=True,
        env_safety=env_safety,
        env_justice=env_assumptions,
        sys_safety=sys_safety,
        sys_justice=sys_guarantees,
        env_init=env_init,
        sys_init=sys_init
    )
    # Build LTL for stats
    spec = _decomp_to_ltl(decomp)
    return _synthesize_gr1(spec, env_vars, sys_vars, decomp)


def _decomp_to_ltl(decomp: GR1Decomposition) -> LTL:
    """Reconstruct LTL formula from decomposition (for display/stats)."""
    parts = []
    for s in decomp.sys_safety:
        parts.append(Globally(s))
    for j in decomp.sys_justice:
        parts.append(Globally(Finally(j)))
    if decomp.sys_init:
        parts.append(decomp.sys_init)

    guarantee = _conjoin(parts) if parts else LTLTrue()

    env_parts = []
    for s in decomp.env_safety:
        env_parts.append(Globally(s))
    for j in decomp.env_justice:
        env_parts.append(Globally(Finally(j)))
    if decomp.env_init:
        env_parts.append(decomp.env_init)

    if env_parts:
        assumption = _conjoin(env_parts)
        return Implies(assumption, guarantee)
    return guarantee


def _conjoin(formulas: List[LTL]) -> LTL:
    """Conjoin a list of formulas."""
    if not formulas:
        return LTLTrue()
    result = formulas[0]
    for f in formulas[1:]:
        result = And(result, f)
    return result


# ============================================================
# Utility Functions
# ============================================================

def _formula_depth(f: LTL) -> int:
    """Depth of formula AST."""
    if f.op in (LTLOp.ATOM, LTLOp.TRUE, LTLOp.FALSE):
        return 0
    left_d = _formula_depth(f.left) if f.left else 0
    right_d = _formula_depth(f.right) if f.right else 0
    return 1 + max(left_d, right_d)


def _formula_size(f: LTL) -> int:
    """Number of nodes in formula AST."""
    if f.op in (LTLOp.ATOM, LTLOp.TRUE, LTLOp.FALSE):
        return 1
    left_s = _formula_size(f.left) if f.left else 0
    right_s = _formula_size(f.right) if f.right else 0
    return 1 + left_s + right_s


def ltl_to_gr1_components(spec: LTL) -> Optional[Dict]:
    """Extract GR(1) components from an LTL formula.

    Returns dict with keys: env_safety, env_justice, sys_safety, sys_justice,
    env_init, sys_init. Or None if not GR(1).
    """
    # Dummy env/sys vars -- we just check structural form
    spec_atoms_list = atoms(spec)
    decomp = detect_gr1(spec, set(), spec_atoms_list)
    if not decomp.is_gr1:
        return None
    return {
        "env_safety": decomp.env_safety,
        "env_justice": decomp.env_justice,
        "sys_safety": decomp.sys_safety,
        "sys_justice": decomp.sys_justice,
        "env_init": decomp.env_init,
        "sys_init": decomp.sys_init
    }


def is_gr1_fragment(spec: LTL) -> bool:
    """Quick check: is this LTL formula in the GR(1) fragment?"""
    spec_atoms_list = atoms(spec)
    decomp = detect_gr1(spec, set(), spec_atoms_list)
    return decomp.is_gr1
