"""V077: LTL Synthesis via GR(1) Reduction.

Reduces LTL synthesis problems to GR(1) games and solves them using V075.
Handles the GR(1)-realizable fragment of LTL:
  - Propositional init conditions
  - Safety: G(phi) where phi is over current+next state variables
  - Liveness: GF(phi) where phi is propositional
  - Response: G(p -> F(q)) encoded as GF(!p | q) liveness
  - Persistence: FG(phi) encoded as safety + liveness

Composes V023 (LTL model checking -- formula AST, parser) with
V075 (GR(1) reactive synthesis -- BDD-based game solving).

Key design decisions:
  - Response patterns use direct GR(1) liveness encoding GF(!p | q) rather
    than auxiliary variable introduction. This avoids state space explosion
    and is compatible with V075's GR(1) fixpoint algorithm.
  - Propositional safety formulas G(phi) are lifted to next-state for system
    variables (since the system controls next-state values in GR(1) games).
  - Safety formulas also generate init constraints (G(phi) implies phi at t=0).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V075_reactive_synthesis'))

from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies, Iff,
    Next, Finally, Globally, Until, Release, WeakUntil,
    LTLTrue, LTLFalse, parse_ltl, atoms, nnf, subformulas
)
from reactive_synthesis import (
    BDD, GR1Spec, SynthResult, SynthesisOutput, MealyMachine,
    GR1Arena, gr1_synthesis, check_realizability, extract_mealy_machine,
    simulate_strategy, verify_controller, make_gr1_game
)
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Callable
from enum import Enum


# ---------------------------------------------------------------------------
# Fragment classification
# ---------------------------------------------------------------------------

class FormulaClass(Enum):
    """Classification of LTL subformulas for GR(1) reduction."""
    INIT = "init"               # Propositional (no temporal ops)
    SAFETY = "safety"           # G(phi) where phi is propositional over curr+next
    LIVENESS = "liveness"       # GF(phi) where phi is propositional
    RESPONSE = "response"       # G(p -> F(q)) where p, q propositional
    PERSISTENCE = "persistence" # FG(phi) where phi is propositional
    UNSUPPORTED = "unsupported" # Not in GR(1) fragment


def is_propositional(f: LTL) -> bool:
    """Check if formula has no temporal operators (pure boolean over atoms)."""
    if f.op in (LTLOp.ATOM, LTLOp.TRUE, LTLOp.FALSE):
        return True
    if f.op in (LTLOp.NOT,):
        return is_propositional(f.left)
    if f.op in (LTLOp.AND, LTLOp.OR, LTLOp.IMPLIES, LTLOp.IFF):
        return is_propositional(f.left) and is_propositional(f.right)
    return False


def is_one_step(f: LTL) -> bool:
    """Check if formula only uses X (next) as temporal -- propositional over curr+next."""
    if f.op in (LTLOp.ATOM, LTLOp.TRUE, LTLOp.FALSE):
        return True
    if f.op == LTLOp.X:
        return is_propositional(f.left)
    if f.op == LTLOp.NOT:
        return is_one_step(f.left)
    if f.op in (LTLOp.AND, LTLOp.OR, LTLOp.IMPLIES, LTLOp.IFF):
        return is_one_step(f.left) and is_one_step(f.right)
    return False


def classify_formula(f: LTL) -> FormulaClass:
    """Classify an LTL formula into GR(1) fragment categories."""
    if is_propositional(f):
        return FormulaClass.INIT

    # G(phi) where phi is one-step -> safety
    if f.op == LTLOp.G and is_one_step(f.left):
        return FormulaClass.SAFETY

    # GF(phi) where phi is propositional -> liveness
    if f.op == LTLOp.G and f.left.op == LTLOp.F and is_propositional(f.left.left):
        return FormulaClass.LIVENESS

    # G(p -> F(q)) where p, q propositional -> response
    if (f.op == LTLOp.G and f.left.op == LTLOp.IMPLIES
            and is_propositional(f.left.left)
            and f.left.right.op == LTLOp.F
            and is_propositional(f.left.right.left)):
        return FormulaClass.RESPONSE

    # FG(phi) where phi propositional -> persistence
    if f.op == LTLOp.F and f.left.op == LTLOp.G and is_propositional(f.left.left):
        return FormulaClass.PERSISTENCE

    return FormulaClass.UNSUPPORTED


def classify_conjunction(f: LTL) -> List[Tuple[FormulaClass, LTL]]:
    """Decompose a conjunction into classified components."""
    if f.op == LTLOp.AND:
        return classify_conjunction(f.left) + classify_conjunction(f.right)
    return [(classify_formula(f), f)]


def is_gr1_fragment(f: LTL) -> bool:
    """Check if formula is in the GR(1)-realizable fragment."""
    parts = classify_conjunction(f)
    return all(cls != FormulaClass.UNSUPPORTED for cls, _ in parts)


# ---------------------------------------------------------------------------
# LTL formula to BDD conversion
# ---------------------------------------------------------------------------

def _prop_to_bdd(bdd: BDD, f: LTL, var_nodes: Dict[str, object]) -> object:
    """Convert propositional LTL formula to BDD node.

    var_nodes: atom name -> BDD node (from bdd.named_var())
    """
    if f.op == LTLOp.TRUE:
        return bdd.TRUE
    if f.op == LTLOp.FALSE:
        return bdd.FALSE
    if f.op == LTLOp.ATOM:
        if f.name not in var_nodes:
            raise ValueError(f"Unknown atom '{f.name}' -- not in variable map")
        return var_nodes[f.name]
    if f.op == LTLOp.NOT:
        return bdd.NOT(_prop_to_bdd(bdd, f.left, var_nodes))
    if f.op == LTLOp.AND:
        return bdd.AND(_prop_to_bdd(bdd, f.left, var_nodes),
                       _prop_to_bdd(bdd, f.right, var_nodes))
    if f.op == LTLOp.OR:
        return bdd.OR(_prop_to_bdd(bdd, f.left, var_nodes),
                      _prop_to_bdd(bdd, f.right, var_nodes))
    if f.op == LTLOp.IMPLIES:
        a = _prop_to_bdd(bdd, f.left, var_nodes)
        b = _prop_to_bdd(bdd, f.right, var_nodes)
        return bdd.OR(bdd.NOT(a), b)
    if f.op == LTLOp.IFF:
        a = _prop_to_bdd(bdd, f.left, var_nodes)
        b = _prop_to_bdd(bdd, f.right, var_nodes)
        return bdd.OR(bdd.AND(a, b), bdd.AND(bdd.NOT(a), bdd.NOT(b)))
    raise ValueError(f"Cannot convert temporal operator {f.op} to BDD")


def _one_step_to_bdd(bdd: BDD, f: LTL, curr_nodes: Dict[str, object],
                     next_nodes: Dict[str, object]) -> object:
    """Convert one-step formula (curr + X(next) refs) to BDD node.

    curr_nodes: atom name -> BDD node for current state
    next_nodes: atom name -> BDD node for next state
    """
    if f.op == LTLOp.TRUE:
        return bdd.TRUE
    if f.op == LTLOp.FALSE:
        return bdd.FALSE
    if f.op == LTLOp.ATOM:
        if f.name not in curr_nodes:
            raise ValueError(f"Unknown atom '{f.name}'")
        return curr_nodes[f.name]
    if f.op == LTLOp.X:
        # Next-state reference
        return _prop_to_bdd(bdd, f.left, next_nodes)
    if f.op == LTLOp.NOT:
        return bdd.NOT(_one_step_to_bdd(bdd, f.left, curr_nodes, next_nodes))
    if f.op in (LTLOp.AND, LTLOp.OR, LTLOp.IMPLIES, LTLOp.IFF):
        a = _one_step_to_bdd(bdd, f.left, curr_nodes, next_nodes)
        b = _one_step_to_bdd(bdd, f.right, curr_nodes, next_nodes)
        if f.op == LTLOp.AND:
            return bdd.AND(a, b)
        if f.op == LTLOp.OR:
            return bdd.OR(a, b)
        if f.op == LTLOp.IMPLIES:
            return bdd.OR(bdd.NOT(a), b)
        # IFF
        return bdd.OR(bdd.AND(a, b), bdd.AND(bdd.NOT(a), bdd.NOT(b)))
    raise ValueError(f"Cannot convert {f.op} in one-step context")


# ---------------------------------------------------------------------------
# Response pattern auxiliary variable construction
# ---------------------------------------------------------------------------

@dataclass
class AuxVariable:
    """Auxiliary variable introduced for response pattern G(p -> Fq)."""
    name: str
    trigger: LTL   # p (the trigger/antecedent)
    target: LTL     # q (the response/consequent)


def response_to_aux(trigger: LTL, target: LTL, aux_name: str) -> AuxVariable:
    """Create auxiliary variable for G(p -> Fq) pattern.

    The response G(p -> Fq) is encoded as:
      Safety: G(aux' <-> ((p | aux) & !q))
        -- aux becomes true when p fires, stays true until q is seen
      Liveness: GF(!aux)
        -- aux must be cleared infinitely often (q must respond to p)
    """
    return AuxVariable(name=aux_name, trigger=trigger, target=target)


# ---------------------------------------------------------------------------
# LTL Synthesis Specification
# ---------------------------------------------------------------------------

@dataclass
class LTLSynthSpec:
    """LTL synthesis specification: assumptions -> guarantees."""
    env_vars: List[str]         # Environment-controlled variables
    sys_vars: List[str]         # System-controlled variables
    assumptions: List[LTL]      # Environment assumptions (conjuncted)
    guarantees: List[LTL]       # System guarantees (conjuncted)

    def all_vars(self) -> List[str]:
        return self.env_vars + self.sys_vars


@dataclass
class ReductionResult:
    """Result of reducing LTL spec to GR(1)."""
    success: bool
    gr1_spec: Optional[GR1Spec] = None
    bdd: Optional[BDD] = None
    aux_vars: List[AuxVariable] = field(default_factory=list)
    env_vars: List[str] = field(default_factory=list)
    sys_vars: List[str] = field(default_factory=list)
    unsupported: List[LTL] = field(default_factory=list)
    classification: Dict[str, List[Tuple[FormulaClass, LTL]]] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class LTLSynthOutput:
    """Output of LTL synthesis."""
    realizable: bool
    strategy: Optional[Dict] = None
    mealy: Optional[MealyMachine] = None
    reduction: Optional[ReductionResult] = None
    synthesis_output: Optional[SynthesisOutput] = None
    statistics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core reduction: LTL -> GR(1)
# ---------------------------------------------------------------------------

def reduce_to_gr1(spec: LTLSynthSpec) -> ReductionResult:
    """Reduce an LTL synthesis specification to a GR(1) game.

    Decomposes assumptions and guarantees into:
      - Init: propositional formulas -> env_init / sys_init
      - Safety: G(phi) -> env_safe / sys_safe
      - Liveness: GF(phi) -> env_live / sys_live
      - Response: G(p -> Fq) -> aux variable + safety + liveness
      - Persistence: FG(phi) -> approximated as safety + liveness
    """
    # Classify all formulas
    assume_parts = []
    for a in spec.assumptions:
        assume_parts.extend(classify_conjunction(a))

    guarantee_parts = []
    for g in spec.guarantees:
        guarantee_parts.extend(classify_conjunction(g))

    # Check for unsupported formulas
    unsupported = [f for cls, f in assume_parts + guarantee_parts
                   if cls == FormulaClass.UNSUPPORTED]
    if unsupported:
        return ReductionResult(
            success=False,
            unsupported=unsupported,
            classification={"assumptions": assume_parts, "guarantees": guarantee_parts},
            error=f"Found {len(unsupported)} formula(s) outside GR(1) fragment"
        )

    # Collect by category
    a_init = [f for cls, f in assume_parts if cls == FormulaClass.INIT]
    a_safety = [f for cls, f in assume_parts if cls == FormulaClass.SAFETY]
    a_liveness = [f for cls, f in assume_parts if cls == FormulaClass.LIVENESS]
    a_response = [f for cls, f in assume_parts if cls == FormulaClass.RESPONSE]
    a_persist = [f for cls, f in assume_parts if cls == FormulaClass.PERSISTENCE]

    g_init = [f for cls, f in guarantee_parts if cls == FormulaClass.INIT]
    g_safety = [f for cls, f in guarantee_parts if cls == FormulaClass.SAFETY]
    g_liveness = [f for cls, f in guarantee_parts if cls == FormulaClass.LIVENESS]
    g_response = [f for cls, f in guarantee_parts if cls == FormulaClass.RESPONSE]
    g_persist = [f for cls, f in guarantee_parts if cls == FormulaClass.PERSISTENCE]

    # Response patterns: G(p -> F(q)) encoded as GF(!p | q) liveness.
    # This is the standard GR(1) encoding (Piterman et al.):
    #   G(p -> F(q)) is approximated by GF(!p | q) -- "infinitely often,
    #   either trigger is false or response is given."
    # No auxiliary variables needed.
    aux_vars = []  # kept for API compat but always empty with direct encoding

    all_env_vars = list(spec.env_vars)
    all_sys_vars = list(spec.sys_vars)

    # Build BDD using named_var (same pattern as make_gr1_game)
    all_vars = all_env_vars + all_sys_vars
    bdd = BDD()

    # Create current and next BDD nodes for each variable
    curr_nodes = {}
    next_nodes = {}
    for v in all_vars:
        curr_nodes[v] = bdd.named_var(v)
        next_nodes[v] = bdd.named_var(v + "'")

    # Build BDD representations
    # --- Init ---
    env_init_bdd = bdd.TRUE
    for f in a_init:
        env_init_bdd = bdd.AND(env_init_bdd, _prop_to_bdd(bdd, f, curr_nodes))

    sys_init_bdd = bdd.TRUE
    for f in g_init:
        sys_init_bdd = bdd.AND(sys_init_bdd, _prop_to_bdd(bdd, f, curr_nodes))

    # --- Safety ---
    # G(phi) means phi holds in ALL states including the initial one.
    # For system guarantees with purely propositional phi (no X/next), we need
    # BOTH an init constraint AND a next-state transition constraint.
    # In GR(1), the system controls next-state sys vars, so propositional
    # safety over sys vars must be lifted to next-state: phi[sys -> sys'].
    env_safe_bdd = bdd.TRUE
    for f in a_safety:
        phi_bdd = _one_step_to_bdd(bdd, f.left, curr_nodes, next_nodes)
        env_safe_bdd = bdd.AND(env_safe_bdd, phi_bdd)
        if is_propositional(f.left):
            env_init_bdd = bdd.AND(env_init_bdd,
                                   _prop_to_bdd(bdd, f.left, curr_nodes))

    sys_safe_bdd = bdd.TRUE
    # Build a "next" map that maps sys vars to their next BDD nodes
    # but keeps env vars at current (env is already committed when system moves)
    sys_next_nodes = {}
    for v in all_env_vars:
        sys_next_nodes[v] = curr_nodes[v]
    for v in all_sys_vars:
        sys_next_nodes[v] = next_nodes[v]

    for f in g_safety:
        if is_propositional(f.left):
            # Propositional safety: sys vars -> next-state (system's choice)
            # Also add to init (must hold at start)
            phi_next = _prop_to_bdd(bdd, f.left, sys_next_nodes)
            sys_safe_bdd = bdd.AND(sys_safe_bdd, phi_next)
            sys_init_bdd = bdd.AND(sys_init_bdd,
                                   _prop_to_bdd(bdd, f.left, curr_nodes))
        else:
            # One-step formula: already has explicit X() for next refs
            phi_bdd = _one_step_to_bdd(bdd, f.left, curr_nodes, next_nodes)
            sys_safe_bdd = bdd.AND(sys_safe_bdd, phi_bdd)

    # Persistence: FG(phi) -- approximate as safety + liveness
    for f in a_persist:
        phi = f.left.left
        env_safe_bdd = bdd.AND(env_safe_bdd, _prop_to_bdd(bdd, phi, curr_nodes))
    for f in g_persist:
        phi = f.left.left
        sys_safe_bdd = bdd.AND(sys_safe_bdd, _prop_to_bdd(bdd, phi, curr_nodes))

    # --- Liveness ---
    env_live_bdds = []
    for f in a_liveness:
        phi = f.left.left  # GF(phi) -> phi is propositional
        env_live_bdds.append(_prop_to_bdd(bdd, phi, curr_nodes))

    # Environment response assumptions: G(p -> Fq) encoded as GF(!p | q)
    for f in a_response:
        p = f.left.left       # trigger
        q = f.left.right.left # target
        p_bdd = _prop_to_bdd(bdd, p, curr_nodes)
        q_bdd = _prop_to_bdd(bdd, q, curr_nodes)
        env_live_bdds.append(bdd.OR(bdd.NOT(p_bdd), q_bdd))

    sys_live_bdds = []
    for f in g_liveness:
        phi = f.left.left
        sys_live_bdds.append(_prop_to_bdd(bdd, phi, curr_nodes))

    # System response guarantees: G(p -> Fq) encoded as GF(!p | q)
    for f in g_response:
        p = f.left.left       # trigger
        q = f.left.right.left # target
        p_bdd = _prop_to_bdd(bdd, p, curr_nodes)
        q_bdd = _prop_to_bdd(bdd, q, curr_nodes)
        sys_live_bdds.append(bdd.OR(bdd.NOT(p_bdd), q_bdd))

    # Persistence liveness
    for f in a_persist:
        phi = f.left.left
        env_live_bdds.append(_prop_to_bdd(bdd, phi, curr_nodes))
    for f in g_persist:
        phi = f.left.left
        sys_live_bdds.append(_prop_to_bdd(bdd, phi, curr_nodes))

    # Build GR1Spec
    gr1 = GR1Spec(
        env_vars=all_env_vars,
        sys_vars=all_sys_vars,
        env_init=env_init_bdd,
        sys_init=sys_init_bdd,
        env_safe=env_safe_bdd,
        sys_safe=sys_safe_bdd,
        env_live=env_live_bdds,
        sys_live=sys_live_bdds,
    )

    return ReductionResult(
        success=True,
        gr1_spec=gr1,
        bdd=bdd,
        aux_vars=aux_vars,
        env_vars=all_env_vars,
        sys_vars=all_sys_vars,
        classification={"assumptions": assume_parts, "guarantees": guarantee_parts},
    )


# ---------------------------------------------------------------------------
# Top-level synthesis API
# ---------------------------------------------------------------------------

def synthesize_ltl(spec: LTLSynthSpec) -> LTLSynthOutput:
    """Synthesize a controller from an LTL specification.

    Takes environment assumptions and system guarantees as LTL formulas,
    reduces to GR(1), and synthesizes a winning strategy if realizable.

    Returns LTLSynthOutput with realizability verdict, strategy, and Mealy machine.
    """
    # Step 1: Reduce to GR(1)
    reduction = reduce_to_gr1(spec)
    if not reduction.success:
        return LTLSynthOutput(
            realizable=False,
            reduction=reduction,
            statistics={"error": reduction.error}
        )

    # Step 2: Run GR(1) synthesis
    synth_out = gr1_synthesis(reduction.bdd, reduction.gr1_spec)

    # Step 3: Extract Mealy machine if realizable
    mealy = None
    if synth_out.result == SynthResult.REALIZABLE:
        mealy = extract_mealy_machine(reduction.bdd, reduction.gr1_spec, synth_out)

    return LTLSynthOutput(
        realizable=(synth_out.result == SynthResult.REALIZABLE),
        strategy=synth_out.strategy,
        mealy=mealy,
        reduction=reduction,
        synthesis_output=synth_out,
        statistics={
            "aux_vars_introduced": len(reduction.aux_vars),
            "total_env_vars": len(reduction.env_vars),
            "total_sys_vars": len(reduction.sys_vars),
            "env_live_count": len(reduction.gr1_spec.env_live),
            "sys_live_count": len(reduction.gr1_spec.sys_live),
            **(synth_out.statistics if synth_out.statistics else {}),
        }
    )


def check_ltl_realizability(spec: LTLSynthSpec) -> bool:
    """Quick realizability check without strategy extraction."""
    reduction = reduce_to_gr1(spec)
    if not reduction.success:
        return False
    return check_realizability(reduction.bdd, reduction.gr1_spec)


# ---------------------------------------------------------------------------
# Convenience: parse-based specification builder
# ---------------------------------------------------------------------------

def make_ltl_spec(env_vars: List[str], sys_vars: List[str],
                  assumptions: List[str], guarantees: List[str]) -> LTLSynthSpec:
    """Build LTL synthesis spec from string formulas.

    Example:
        spec = make_ltl_spec(
            env_vars=["req"],
            sys_vars=["grant"],
            assumptions=["G(F(req))"],
            guarantees=["G(req -> F(grant))", "G(!req -> !grant)"]
        )
    """
    return LTLSynthSpec(
        env_vars=env_vars,
        sys_vars=sys_vars,
        assumptions=[parse_ltl(a) for a in assumptions],
        guarantees=[parse_ltl(g) for g in guarantees],
    )


def synthesize_from_strings(env_vars: List[str], sys_vars: List[str],
                            assumptions: List[str],
                            guarantees: List[str]) -> LTLSynthOutput:
    """One-shot synthesis from string formulas."""
    spec = make_ltl_spec(env_vars, sys_vars, assumptions, guarantees)
    return synthesize_ltl(spec)


# ---------------------------------------------------------------------------
# Simulation and verification
# ---------------------------------------------------------------------------

def simulate_ltl_controller(output: LTLSynthOutput,
                            env_trace: List[Dict[str, bool]],
                            max_steps: int = 20) -> Optional[List[Dict[str, bool]]]:
    """Simulate synthesized controller against an environment trace.

    Returns list of full state dicts (env + sys vars) at each step,
    or None if not realizable.
    """
    if not output.realizable or output.reduction is None:
        return None
    return simulate_strategy(
        output.reduction.bdd, output.reduction.gr1_spec,
        output.synthesis_output, env_trace, max_steps
    )


def verify_ltl_controller(output: LTLSynthOutput) -> Optional[Dict]:
    """Verify synthesized controller satisfies the GR(1) spec.

    Checks: init in winning, winning closed under CPre, strategy safety.
    """
    if not output.realizable or output.reduction is None:
        return None
    return verify_controller(
        output.reduction.bdd, output.reduction.gr1_spec,
        output.synthesis_output
    )


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def analyze_spec(spec: LTLSynthSpec) -> Dict:
    """Analyze an LTL spec: classify formulas, check GR(1) membership."""
    assume_parts = []
    for a in spec.assumptions:
        assume_parts.extend(classify_conjunction(a))

    guarantee_parts = []
    for g in spec.guarantees:
        guarantee_parts.extend(classify_conjunction(g))

    all_parts = assume_parts + guarantee_parts
    in_fragment = all(cls != FormulaClass.UNSUPPORTED for cls, _ in all_parts)

    counts = {}
    for cls, _ in all_parts:
        counts[cls.value] = counts.get(cls.value, 0) + 1

    return {
        "in_gr1_fragment": in_fragment,
        "assumption_count": len(assume_parts),
        "guarantee_count": len(guarantee_parts),
        "formula_counts": counts,
        "assumptions": [(cls.value, str(f)) for cls, f in assume_parts],
        "guarantees": [(cls.value, str(f)) for cls, f in guarantee_parts],
        "unsupported": [str(f) for cls, f in all_parts if cls == FormulaClass.UNSUPPORTED],
        "response_patterns": sum(1 for cls, _ in all_parts if cls == FormulaClass.RESPONSE),
        "aux_vars_needed": sum(1 for cls, _ in all_parts if cls == FormulaClass.RESPONSE),
    }


# ---------------------------------------------------------------------------
# Example synthesis problems
# ---------------------------------------------------------------------------

def synthesize_arbiter_ltl(n_clients: int = 2) -> LTLSynthOutput:
    """Synthesize mutual exclusion arbiter from LTL spec.

    Env vars: req_0, req_1, ...
    Sys vars: grant_0, grant_1, ...

    Assumptions:
      - GF(req_i) for each client (fairness: each client requests infinitely often)

    Guarantees:
      - G(req_i -> F(grant_i)) for each client (response: every request gets granted)
      - G(!(grant_i & grant_j)) for i != j (safety: mutual exclusion)
      - G(!req_i -> !grant_i) for each client (safety: no spurious grants)
    """
    env_vars = [f"req_{i}" for i in range(n_clients)]
    sys_vars = [f"grant_{i}" for i in range(n_clients)]

    assumptions = [f"G(F(req_{i}))" for i in range(n_clients)]

    guarantees = []
    # Response: every request eventually granted
    for i in range(n_clients):
        guarantees.append(f"G(req_{i} -> F(grant_{i}))")
    # Mutual exclusion
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            guarantees.append(f"G(!(grant_{i} & grant_{j}))")
    # No spurious grants
    for i in range(n_clients):
        guarantees.append(f"G(!req_{i} -> !grant_{i})")

    return synthesize_from_strings(env_vars, sys_vars, assumptions, guarantees)


def synthesize_traffic_light_ltl() -> LTLSynthOutput:
    """Synthesize traffic light controller from LTL spec.

    Env vars: car (car waiting at intersection)
    Sys vars: green (light is green)

    Assumptions:
      - GF(car) -- cars keep arriving

    Guarantees:
      - G(car -> F(green)) -- every waiting car eventually gets green
      - GF(!green) -- light isn't always green (fairness to cross traffic)
    """
    return synthesize_from_strings(
        env_vars=["car"],
        sys_vars=["green"],
        assumptions=["G(F(car))"],
        guarantees=["G(car -> F(green))", "G(F(!green))"]
    )


def synthesize_buffer_ltl() -> LTLSynthOutput:
    """Synthesize single-cell buffer controller.

    Env vars: write (write request), data_in (input data bit)
    Sys vars: read_ready (data available), data_out (output data bit)

    Guarantees:
      - G(write -> F(read_ready)) -- writes eventually become readable
      - G(!write -> X(!read_ready)) -- no write means no new data next step
    """
    return synthesize_from_strings(
        env_vars=["write", "data_in"],
        sys_vars=["read_ready", "data_out"],
        assumptions=["G(F(write))"],
        guarantees=[
            "G(write -> F(read_ready))",
            "G(!write -> X(!read_ready))",
        ]
    )
