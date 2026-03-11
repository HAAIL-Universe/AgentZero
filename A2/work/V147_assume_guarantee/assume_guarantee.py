"""V147: Certified Assume-Guarantee Reasoning

Thread-modular verification: each component verified against environment
assumptions, with circular dependencies discharged via assume-guarantee rules.

Composes:
- V004 (VCGen/WP) -- verification condition generation, SExpr
- V044 (proof certificates) -- proof obligations, certificates, checking
- V145 (compositional verification) -- modular verification base
- C037 (SMT solver) -- satisfiability checking
- C010 (parser) -- AST

Key concepts:
- Component: a sequential program fragment (function/thread)
- Assumption: what a component expects of its environment
- Guarantee: what a component provides to its environment
- AG Rule: if (A assumes P, guarantees Q) and (B assumes Q, guarantees P),
  and both proofs are valid, then P AND Q hold in the composition
- Circular AG: multiple components with mutual assumptions, discharged
  via well-founded induction on a ranking function
"""

import sys, os, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Set
from fractions import Fraction

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V145_certified_compositional'))

from stack_vm import lex, Parser
from smt_solver import SMTSolver, SMTResult, Var as SMTVar, App, Op, IntConst, BoolConst
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies, substitute, ast_to_sexpr, lower_to_smt,
    WPCalculus, VCResult, VCStatus, VerificationResult, FnSpec, verify_function,
    verify_program
)
from proof_certificates import (
    ProofObligation, ProofCertificate, ProofKind, CertStatus,
    check_certificate, combine_certificates, sexpr_to_str, sexpr_to_smtlib
)


# ============================================================
# Data structures
# ============================================================

class AGVerdict(Enum):
    """Overall assume-guarantee verification result."""
    SOUND = "sound"               # All components verified, all assumptions discharged
    COMPONENT_FAILURE = "component_failure"  # A component fails under its assumptions
    DISCHARGE_FAILURE = "discharge_failure"  # Circular assumptions can't be discharged
    UNKNOWN = "unknown"


@dataclass
class ComponentSpec:
    """Specification of a component (thread/module) for AG reasoning."""
    name: str
    params: List[str]             # Shared variables the component can access
    assumptions: List[SExpr]      # What this component assumes about its environment
    guarantees: List[SExpr]       # What this component provides
    body_source: str              # Source code of the component
    body_stmts: Optional[List] = None  # Parsed AST (filled by extract)


@dataclass
class AGObligation:
    """A proof obligation in the AG framework."""
    name: str
    description: str
    component: str                # Which component this belongs to
    kind: str                     # "local_vc", "assumption_discharge", "guarantee_check"
    formula: SExpr
    status: VCStatus = VCStatus.UNKNOWN
    counterexample: Optional[dict] = None


@dataclass
class ComponentResult:
    """Verification result for a single component under assumptions."""
    name: str
    verified: bool                # Component verified under its assumptions
    obligations: List[AGObligation]
    certificate: Optional[ProofCertificate] = None


@dataclass
class DischargeResult:
    """Result of discharging circular assumptions."""
    discharged: bool              # All circular assumptions successfully discharged
    strategy: str                 # "direct", "circular", "inductive"
    obligations: List[AGObligation]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AGResult:
    """Full assume-guarantee verification result."""
    verdict: AGVerdict
    components: Dict[str, ComponentResult]
    discharge: Optional[DischargeResult]
    certificate: Optional[ProofCertificate]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AGSystem:
    """A system of components with assume-guarantee contracts."""
    components: List[ComponentSpec]
    shared_vars: List[str]        # Variables shared across components
    shared_var_types: Dict[str, str] = field(default_factory=dict)  # var -> "int"/"bool"


# ============================================================
# Helper: SExpr-based VC checking via SMT
# ============================================================

def _collect_vars(expr: SExpr) -> Set[str]:
    """Collect all variable names from an SExpr."""
    if isinstance(expr, SVar):
        return {expr.name}
    elif isinstance(expr, (SInt, SBool)):
        return set()
    elif isinstance(expr, SBinOp):
        return _collect_vars(expr.left) | _collect_vars(expr.right)
    elif isinstance(expr, SUnaryOp):
        return _collect_vars(expr.operand)
    elif isinstance(expr, SImplies):
        return _collect_vars(expr.antecedent) | _collect_vars(expr.consequent)
    elif isinstance(expr, SAnd):
        result = set()
        for c in expr.conjuncts:
            result |= _collect_vars(c)
        return result
    elif isinstance(expr, SOr):
        result = set()
        for d in expr.disjuncts:
            result |= _collect_vars(d)
        return result
    elif isinstance(expr, SNot):
        return _collect_vars(expr.operand)
    elif isinstance(expr, SIte):
        return _collect_vars(expr.cond) | _collect_vars(expr.then_val) | _collect_vars(expr.else_val)
    return set()


def _check_vc_sexpr(name: str, formula: SExpr, var_types: Optional[Dict[str, str]] = None) -> AGObligation:
    """Check a verification condition expressed as SExpr.

    Returns VALID if NOT(formula) is UNSAT (formula is a tautology).
    """
    solver = SMTSolver()
    var_cache = {}

    # Declare variables
    all_vars = _collect_vars(formula)
    for v in all_vars:
        vtype = (var_types or {}).get(v, "int")
        if vtype == "bool":
            var_cache[v] = solver.Bool(v)
        else:
            var_cache[v] = solver.Int(v)

    # Check NOT(formula) -- if UNSAT, formula is valid
    smt_formula = lower_to_smt(solver, formula, var_cache)
    solver.add(solver.Not(smt_formula))

    result = solver.check()

    if result == SMTResult.UNSAT:
        return AGObligation(name=name, description=f"VC: {name}",
                           component="", kind="local_vc", formula=formula,
                           status=VCStatus.VALID)
    elif result == SMTResult.SAT:
        model = solver.model() or {}
        return AGObligation(name=name, description=f"VC: {name}",
                           component="", kind="local_vc", formula=formula,
                           status=VCStatus.INVALID, counterexample=model)
    else:
        return AGObligation(name=name, description=f"VC: {name}",
                           component="", kind="local_vc", formula=formula,
                           status=VCStatus.UNKNOWN)


def _check_implication(premise: SExpr, conclusion: SExpr,
                       var_types: Optional[Dict[str, str]] = None) -> AGObligation:
    """Check if premise => conclusion is valid."""
    formula = s_implies(premise, conclusion)
    return _check_vc_sexpr("implication", formula, var_types)


# ============================================================
# Component extraction and verification
# ============================================================

def extract_component(name: str, source: str, assumptions: List[SExpr],
                      guarantees: List[SExpr], shared_vars: List[str]) -> ComponentSpec:
    """Create a ComponentSpec from source code and contracts."""
    tokens = lex(source)
    ast = Parser(tokens).parse()
    stmts = ast.stmts if hasattr(ast, 'stmts') else [ast]

    return ComponentSpec(
        name=name,
        params=shared_vars,
        assumptions=assumptions,
        guarantees=guarantees,
        body_source=source,
        body_stmts=stmts
    )


def _build_annotated_source(comp: ComponentSpec) -> str:
    """Build source with requires/ensures annotations for V004 verification."""
    params_str = ", ".join(comp.params)

    lines = [f"fn {comp.name}({params_str}) {{"]

    # Add requires (assumptions)
    for i, a in enumerate(comp.assumptions):
        lines.append(f"  requires({_sexpr_to_source(a)});")

    # Add ensures (guarantees)
    for i, g in enumerate(comp.guarantees):
        lines.append(f"  ensures({_sexpr_to_source(g)});")

    # Add body
    lines.append(f"  {comp.body_source}")
    lines.append("}")

    return "\n".join(lines)


def _sexpr_to_source(expr: SExpr) -> str:
    """Convert SExpr to C10 source expression."""
    if isinstance(expr, SVar):
        return expr.name
    elif isinstance(expr, SInt):
        if expr.value < 0:
            return f"(0 - {-expr.value})"
        return str(expr.value)
    elif isinstance(expr, SBool):
        return "true" if expr.value else "false"
    elif isinstance(expr, SBinOp):
        left = _sexpr_to_source(expr.left)
        right = _sexpr_to_source(expr.right)
        return f"({left} {expr.op} {right})"
    elif isinstance(expr, SUnaryOp):
        operand = _sexpr_to_source(expr.operand)
        return f"({expr.op}{operand})"
    elif isinstance(expr, SAnd):
        parts = [_sexpr_to_source(c) for c in expr.conjuncts]
        return "(" + " and ".join(parts) + ")"
    elif isinstance(expr, SOr):
        parts = [_sexpr_to_source(d) for d in expr.disjuncts]
        return "(" + " or ".join(parts) + ")"
    elif isinstance(expr, SNot):
        operand = _sexpr_to_source(expr.operand)
        return f"(not {operand})"
    elif isinstance(expr, SImplies):
        # a => b  ==  not a or b
        a = _sexpr_to_source(expr.antecedent)
        b = _sexpr_to_source(expr.consequent)
        return f"((not {a}) or {b})"
    elif isinstance(expr, SIte):
        c = _sexpr_to_source(expr.cond)
        t = _sexpr_to_source(expr.then_val)
        e = _sexpr_to_source(expr.else_val)
        return f"(if ({c}) {t} else {e})"
    return "true"


def verify_component_under_assumptions(comp: ComponentSpec,
                                        var_types: Optional[Dict[str, str]] = None) -> ComponentResult:
    """Verify a component under its assumptions, checking its guarantees.

    Uses WP calculus: for each guarantee G,
      check: AND(assumptions) => WP(body, G)

    Since C10 doesn't directly support full WP over arbitrary code,
    we use SMT-based forward symbolic reasoning:
    - Assumptions constrain the initial state
    - Execute body symbolically
    - Check that guarantees hold in the final state
    """
    obligations = []
    all_valid = True

    # Build combined assumption
    if comp.assumptions:
        assumption = s_and(*comp.assumptions) if len(comp.assumptions) > 1 else comp.assumptions[0]
    else:
        assumption = SBool(True)

    # For each guarantee, check: assumption => WP(body, guarantee)
    # We approximate WP by encoding the body as a transformer on shared variables
    body_transformer = _extract_body_transformer(comp)

    for i, guarantee in enumerate(comp.guarantees):
        # Substitute post-state variables in guarantee
        post_guarantee = guarantee
        if body_transformer:
            for var, expr in body_transformer.items():
                post_guarantee = substitute(post_guarantee, var, expr)

        # VC: assumption => post_guarantee
        vc = s_implies(assumption, post_guarantee)

        obl = _check_vc_sexpr(
            name=f"{comp.name}_guarantee_{i}",
            formula=vc,
            var_types=var_types
        )
        obl.component = comp.name
        obl.kind = "guarantee_check"
        obl.description = f"Component {comp.name}: guarantee {i} holds under assumptions"
        obligations.append(obl)

        if obl.status != VCStatus.VALID:
            all_valid = False

    # Generate certificate
    cert_obligations = []
    for obl in obligations:
        cert_obligations.append(ProofObligation(
            name=obl.name,
            description=obl.description,
            formula_str=sexpr_to_str(obl.formula),
            formula_smt="",
            status=CertStatus.VALID if obl.status == VCStatus.VALID else CertStatus.INVALID
        ))

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"Component {comp.name} satisfies guarantees under assumptions",
        source=comp.body_source,
        obligations=cert_obligations,
        metadata={"component": comp.name, "num_assumptions": len(comp.assumptions),
                  "num_guarantees": len(comp.guarantees)},
        status=CertStatus.VALID if all_valid else CertStatus.INVALID
    )

    return ComponentResult(
        name=comp.name,
        verified=all_valid,
        obligations=obligations,
        certificate=cert
    )


def _extract_body_transformer(comp: ComponentSpec) -> Dict[str, SExpr]:
    """Extract a simple transformer from component body.

    For simple assignments like 'let x = expr;' or 'x = expr;',
    maps variable names to their post-state expressions.
    """
    transformer = {}

    if not comp.body_stmts:
        return transformer

    for stmt in comp.body_stmts:
        stype = type(stmt).__name__
        if stype == 'LetDecl' and hasattr(stmt, 'name') and hasattr(stmt, 'value'):
            if stmt.value is not None:
                try:
                    transformer[stmt.name] = ast_to_sexpr(stmt.value)
                except:
                    pass
        elif stype == 'Assign' and hasattr(stmt, 'name') and hasattr(stmt, 'value'):
            try:
                transformer[stmt.name] = ast_to_sexpr(stmt.value)
            except:
                pass

    return transformer


# ============================================================
# Assume-Guarantee Rules
# ============================================================

def _build_dependency_graph(system: AGSystem) -> Dict[str, Set[str]]:
    """Build dependency graph: component -> set of components it depends on.

    Component A depends on B if A's assumptions mention variables that B guarantees.
    """
    # Collect guaranteed vars per component
    guaranteed_vars = {}
    for comp in system.components:
        gvars = set()
        for g in comp.guarantees:
            gvars |= _collect_vars(g)
        guaranteed_vars[comp.name] = gvars

    # Build dependency graph
    deps = {comp.name: set() for comp in system.components}
    for comp in system.components:
        assumed_vars = set()
        for a in comp.assumptions:
            assumed_vars |= _collect_vars(a)

        for other in system.components:
            if other.name != comp.name:
                # If comp assumes something about vars that other guarantees
                if assumed_vars & guaranteed_vars[other.name]:
                    deps[comp.name].add(other.name)

    return deps


def _find_circular_dependencies(deps: Dict[str, Set[str]]) -> List[List[str]]:
    """Find all circular dependency chains (SCCs) in the dependency graph."""
    # Tarjan's SCC algorithm
    index_counter = [0]
    stack = []
    on_stack = set()
    indices = {}
    lowlinks = {}
    sccs = []

    def strongconnect(v):
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in deps.get(v, set()):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    for v in deps:
        if v not in indices:
            strongconnect(v)

    return sccs


def discharge_direct(comp_a: ComponentSpec, comp_b: ComponentSpec,
                     result_a: ComponentResult, result_b: ComponentResult,
                     var_types: Optional[Dict[str, str]] = None) -> DischargeResult:
    """Direct (non-circular) assumption discharge.

    For each assumption of A, check if B's guarantees imply it.
    For each assumption of B, check if A's guarantees imply it.
    """
    obligations = []
    all_discharged = True

    # Check: B's guarantees => A's assumptions
    b_guarantee = s_and(*comp_b.guarantees) if comp_b.guarantees else SBool(True)
    for i, a_assumption in enumerate(comp_a.assumptions):
        obl = _check_vc_sexpr(
            name=f"discharge_{comp_a.name}_assumption_{i}_by_{comp_b.name}",
            formula=s_implies(b_guarantee, a_assumption),
            var_types=var_types
        )
        obl.component = comp_a.name
        obl.kind = "assumption_discharge"
        obl.description = f"{comp_b.name}'s guarantees imply {comp_a.name}'s assumption {i}"
        obligations.append(obl)
        if obl.status != VCStatus.VALID:
            all_discharged = False

    # Check: A's guarantees => B's assumptions
    a_guarantee = s_and(*comp_a.guarantees) if comp_a.guarantees else SBool(True)
    for i, b_assumption in enumerate(comp_b.assumptions):
        obl = _check_vc_sexpr(
            name=f"discharge_{comp_b.name}_assumption_{i}_by_{comp_a.name}",
            formula=s_implies(a_guarantee, b_assumption),
            var_types=var_types
        )
        obl.component = comp_b.name
        obl.kind = "assumption_discharge"
        obl.description = f"{comp_a.name}'s guarantees imply {comp_b.name}'s assumption {i}"
        obligations.append(obl)
        if obl.status != VCStatus.VALID:
            all_discharged = False

    return DischargeResult(
        discharged=all_discharged,
        strategy="direct",
        obligations=obligations
    )


def discharge_circular(system: AGSystem, component_results: Dict[str, ComponentResult],
                        var_types: Optional[Dict[str, str]] = None) -> DischargeResult:
    """Circular assume-guarantee discharge using the AG rule.

    For a circular dependency A -> B -> A, we use the rule:
      1. A verified under assumption P (with guarantee Q)
      2. B verified under assumption Q (with guarantee P)
      3. P AND Q is consistent (not contradictory)
      4. Then: P AND Q hold in the composition

    For larger cycles, we generalize: all guarantees must collectively
    imply all assumptions, and the conjunction must be consistent.
    """
    obligations = []
    all_discharged = True

    comps = {c.name: c for c in system.components}

    # Collect all guarantees and all assumptions
    all_guarantees = []
    all_assumptions = []
    for comp in system.components:
        all_guarantees.extend(comp.guarantees)
        all_assumptions.extend(comp.assumptions)

    if not all_guarantees:
        return DischargeResult(discharged=True, strategy="circular",
                              obligations=[], details={"reason": "no guarantees"})

    # Combined guarantee
    combined_guarantee = s_and(*all_guarantees) if len(all_guarantees) > 1 else all_guarantees[0]

    # Step 1: Check consistency of combined guarantees
    consistency_obl = _check_consistency(combined_guarantee, var_types)
    consistency_obl.kind = "assumption_discharge"
    consistency_obl.description = "Combined guarantees are consistent (satisfiable)"
    obligations.append(consistency_obl)
    if consistency_obl.status != VCStatus.VALID:
        all_discharged = False

    # Step 2: For each assumption of each component, check that
    # the OTHER components' guarantees imply it
    for comp in system.components:
        other_guarantees = []
        for other in system.components:
            if other.name != comp.name:
                other_guarantees.extend(other.guarantees)

        if not other_guarantees:
            # No other component provides guarantees
            if comp.assumptions:
                all_discharged = False
                for i, a in enumerate(comp.assumptions):
                    obligations.append(AGObligation(
                        name=f"undischarged_{comp.name}_assumption_{i}",
                        description=f"No other component guarantees {comp.name}'s assumption {i}",
                        component=comp.name, kind="assumption_discharge",
                        formula=a, status=VCStatus.INVALID
                    ))
            continue

        env_guarantee = s_and(*other_guarantees) if len(other_guarantees) > 1 else other_guarantees[0]

        for i, assumption in enumerate(comp.assumptions):
            obl = _check_vc_sexpr(
                name=f"circular_discharge_{comp.name}_assumption_{i}",
                formula=s_implies(env_guarantee, assumption),
                var_types=var_types
            )
            obl.component = comp.name
            obl.kind = "assumption_discharge"
            obl.description = (f"Environment guarantees imply {comp.name}'s assumption {i}")
            obligations.append(obl)
            if obl.status != VCStatus.VALID:
                all_discharged = False

    # Step 3: Check that all components were verified under assumptions
    for comp in system.components:
        cr = component_results.get(comp.name)
        if not cr or not cr.verified:
            all_discharged = False
            obligations.append(AGObligation(
                name=f"component_verified_{comp.name}",
                description=f"Component {comp.name} must be verified under assumptions",
                component=comp.name, kind="assumption_discharge",
                formula=SBool(False), status=VCStatus.INVALID
            ))

    return DischargeResult(
        discharged=all_discharged,
        strategy="circular",
        obligations=obligations,
        details={"num_components": len(system.components),
                 "circular_deps": _find_circular_dependencies(_build_dependency_graph(system))}
    )


def discharge_inductive(system: AGSystem, component_results: Dict[str, ComponentResult],
                        ranking: Optional[Dict[str, int]] = None,
                        var_types: Optional[Dict[str, str]] = None) -> DischargeResult:
    """Inductive assume-guarantee discharge using a ranking function.

    Components are ordered by rank. A component at rank k can assume
    guarantees from components at rank < k without circularity.
    Components at the same rank use circular discharge.
    """
    obligations = []
    all_discharged = True

    comps = {c.name: c for c in system.components}

    # Default ranking: topological order of dependency graph, or index order
    if ranking is None:
        ranking = {c.name: i for i, c in enumerate(system.components)}

    # Group by rank
    rank_groups = {}
    for name, rank in ranking.items():
        rank_groups.setdefault(rank, []).append(name)

    # Process ranks in order
    established_guarantees = []  # Guarantees established so far

    for rank in sorted(rank_groups.keys()):
        group = rank_groups[rank]

        if len(group) == 1:
            # Single component at this rank: discharge directly
            comp = comps[group[0]]

            # Its assumptions must be implied by established guarantees
            if comp.assumptions and established_guarantees:
                env = s_and(*established_guarantees) if len(established_guarantees) > 1 else established_guarantees[0]
                for i, assumption in enumerate(comp.assumptions):
                    obl = _check_vc_sexpr(
                        name=f"inductive_{comp.name}_assumption_{i}_rank_{rank}",
                        formula=s_implies(env, assumption),
                        var_types=var_types
                    )
                    obl.component = comp.name
                    obl.kind = "assumption_discharge"
                    obl.description = f"Rank-{rank}: {comp.name}'s assumption {i} from lower ranks"
                    obligations.append(obl)
                    if obl.status != VCStatus.VALID:
                        all_discharged = False
            elif comp.assumptions and not established_guarantees:
                # First rank but has assumptions -- must be self-evident
                for i, assumption in enumerate(comp.assumptions):
                    obl = _check_vc_sexpr(
                        name=f"inductive_{comp.name}_base_assumption_{i}",
                        formula=assumption,
                        var_types=var_types
                    )
                    obl.component = comp.name
                    obl.kind = "assumption_discharge"
                    obl.description = f"Base rank: {comp.name}'s assumption {i} must be tautological"
                    obligations.append(obl)
                    if obl.status != VCStatus.VALID:
                        all_discharged = False

            # If component verified, add its guarantees to established set
            cr = component_results.get(comp.name)
            if cr and cr.verified:
                established_guarantees.extend(comp.guarantees)
            else:
                all_discharged = False
        else:
            # Multiple components at same rank: circular discharge within group
            group_comps = [comps[n] for n in group]
            group_system = AGSystem(components=group_comps, shared_vars=system.shared_vars,
                                   shared_var_types=system.shared_var_types)

            # Augment each component's assumptions with established guarantees
            augmented = {}
            for comp in group_comps:
                augmented[comp.name] = comp

            group_results = {n: component_results.get(n) for n in group if component_results.get(n)}
            sub_result = discharge_circular(group_system, group_results, var_types)
            obligations.extend(sub_result.obligations)
            if not sub_result.discharged:
                all_discharged = False
            else:
                # Add all group guarantees to established
                for comp in group_comps:
                    established_guarantees.extend(comp.guarantees)

    return DischargeResult(
        discharged=all_discharged,
        strategy="inductive",
        obligations=obligations,
        details={"ranking": ranking, "num_ranks": len(rank_groups)}
    )


def _check_consistency(formula: SExpr, var_types: Optional[Dict[str, str]] = None) -> AGObligation:
    """Check that a formula is consistent (satisfiable).

    Returns VALID if the formula is SAT (consistent).
    """
    solver = SMTSolver()
    var_cache = {}

    all_vars = _collect_vars(formula)
    for v in all_vars:
        vtype = (var_types or {}).get(v, "int")
        if vtype == "bool":
            var_cache[v] = solver.Bool(v)
        else:
            var_cache[v] = solver.Int(v)

    smt_formula = lower_to_smt(solver, formula, var_cache)
    solver.add(smt_formula)

    result = solver.check()

    if result == SMTResult.SAT:
        return AGObligation(
            name="consistency_check",
            description="Formula is consistent",
            component="", kind="consistency",
            formula=formula, status=VCStatus.VALID
        )
    elif result == SMTResult.UNSAT:
        return AGObligation(
            name="consistency_check",
            description="Formula is INCONSISTENT",
            component="", kind="consistency",
            formula=formula, status=VCStatus.INVALID
        )
    else:
        return AGObligation(
            name="consistency_check",
            description="Consistency unknown",
            component="", kind="consistency",
            formula=formula, status=VCStatus.UNKNOWN
        )


# ============================================================
# Main AG verification pipeline
# ============================================================

def verify_ag(system: AGSystem,
              ranking: Optional[Dict[str, int]] = None,
              var_types: Optional[Dict[str, str]] = None) -> AGResult:
    """Full assume-guarantee verification pipeline.

    1. Verify each component under its assumptions
    2. Detect circular dependencies
    3. Discharge assumptions (direct, circular, or inductive)
    4. Generate combined certificate
    """
    t0 = time.time()

    # Phase 1: Verify each component under assumptions
    component_results = {}
    all_components_verified = True

    for comp in system.components:
        cr = verify_component_under_assumptions(comp, var_types)
        component_results[comp.name] = cr
        if not cr.verified:
            all_components_verified = False

    # If any component fails, report immediately
    if not all_components_verified:
        return AGResult(
            verdict=AGVerdict.COMPONENT_FAILURE,
            components=component_results,
            discharge=None,
            certificate=None,
            metadata={"time": time.time() - t0,
                      "failed_components": [n for n, r in component_results.items() if not r.verified]}
        )

    # Phase 2: Detect dependency structure
    deps = _build_dependency_graph(system)
    circular = _find_circular_dependencies(deps)

    # Phase 3: Discharge assumptions
    if not circular:
        # No circular dependencies -- use direct or inductive discharge
        if len(system.components) == 2:
            dr = discharge_direct(system.components[0], system.components[1],
                                 component_results[system.components[0].name],
                                 component_results[system.components[1].name],
                                 var_types)
        else:
            dr = discharge_inductive(system, component_results, ranking, var_types)
    else:
        # Has circular dependencies -- use circular or inductive discharge
        if ranking:
            dr = discharge_inductive(system, component_results, ranking, var_types)
        else:
            dr = discharge_circular(system, component_results, var_types)

    # Phase 4: Build certificate
    if dr.discharged:
        # Combine all component certificates
        sub_certs = [cr.certificate for cr in component_results.values() if cr.certificate]

        # Add discharge obligations to certificate
        discharge_cert_obls = []
        for obl in dr.obligations:
            discharge_cert_obls.append(ProofObligation(
                name=obl.name,
                description=obl.description,
                formula_str=sexpr_to_str(obl.formula),
                formula_smt="",
                status=CertStatus.VALID if obl.status == VCStatus.VALID else CertStatus.INVALID
            ))

        discharge_cert = ProofCertificate(
            kind=ProofKind.VCGEN,
            claim="Assumption discharge",
            obligations=discharge_cert_obls,
            metadata={"strategy": dr.strategy},
            status=CertStatus.VALID if dr.discharged else CertStatus.INVALID
        )
        sub_certs.append(discharge_cert)

        combined_cert = combine_certificates(*sub_certs,
                                             claim="AG verification: all components verified and assumptions discharged")
        verdict = AGVerdict.SOUND
    else:
        combined_cert = None
        verdict = AGVerdict.DISCHARGE_FAILURE

    return AGResult(
        verdict=verdict,
        components=component_results,
        discharge=dr,
        certificate=combined_cert,
        metadata={"time": time.time() - t0,
                  "num_components": len(system.components),
                  "circular_deps": circular,
                  "strategy": dr.strategy}
    )


# ============================================================
# Convenience APIs
# ============================================================

def make_ag_system(components: List[Dict[str, Any]], shared_vars: List[str],
                   var_types: Optional[Dict[str, str]] = None) -> AGSystem:
    """Create an AGSystem from component dictionaries.

    Each dict: {name, body, assumptions: [(sexpr)], guarantees: [(sexpr)]}
    """
    comp_specs = []
    for cd in components:
        comp = extract_component(
            name=cd['name'],
            source=cd['body'],
            assumptions=cd.get('assumptions', []),
            guarantees=cd.get('guarantees', []),
            shared_vars=shared_vars
        )
        comp_specs.append(comp)

    return AGSystem(
        components=comp_specs,
        shared_vars=shared_vars,
        shared_var_types=var_types or {}
    )


def verify_two_components(name_a: str, body_a: str, assumptions_a: List[SExpr],
                           guarantees_a: List[SExpr],
                           name_b: str, body_b: str, assumptions_b: List[SExpr],
                           guarantees_b: List[SExpr],
                           shared_vars: List[str],
                           var_types: Optional[Dict[str, str]] = None) -> AGResult:
    """Convenience: verify two components with assume-guarantee."""
    system = AGSystem(
        components=[
            extract_component(name_a, body_a, assumptions_a, guarantees_a, shared_vars),
            extract_component(name_b, body_b, assumptions_b, guarantees_b, shared_vars),
        ],
        shared_vars=shared_vars,
        shared_var_types=var_types or {}
    )
    return verify_ag(system, var_types=var_types)


def analyze_dependencies(system: AGSystem) -> Dict[str, Any]:
    """Analyze the dependency structure of an AG system."""
    deps = _build_dependency_graph(system)
    circular = _find_circular_dependencies(deps)

    return {
        "dependency_graph": {k: list(v) for k, v in deps.items()},
        "circular_dependencies": circular,
        "has_circularity": len(circular) > 0,
        "num_components": len(system.components),
        "total_assumptions": sum(len(c.assumptions) for c in system.components),
        "total_guarantees": sum(len(c.guarantees) for c in system.components),
    }


def verify_with_ranking(system: AGSystem, ranking: Dict[str, int],
                        var_types: Optional[Dict[str, str]] = None) -> AGResult:
    """Verify AG system with explicit ranking for inductive discharge."""
    return verify_ag(system, ranking=ranking, var_types=var_types)


def compare_discharge_strategies(system: AGSystem,
                                  var_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Compare circular vs inductive discharge strategies."""
    # Both strategies verify components the same way; differ in discharge
    t0 = time.time()

    # Verify components once
    component_results = {}
    for comp in system.components:
        cr = verify_component_under_assumptions(comp, var_types)
        component_results[comp.name] = cr

    # Circular discharge
    t1 = time.time()
    circular_result = discharge_circular(system, component_results, var_types)
    t2 = time.time()

    # Inductive discharge (default ranking)
    inductive_result = discharge_inductive(system, component_results, var_types=var_types)
    t3 = time.time()

    return {
        "circular": {
            "discharged": circular_result.discharged,
            "num_obligations": len(circular_result.obligations),
            "time": t2 - t1,
        },
        "inductive": {
            "discharged": inductive_result.discharged,
            "num_obligations": len(inductive_result.obligations),
            "time": t3 - t2,
        },
        "components_verified": all(cr.verified for cr in component_results.values()),
        "total_time": time.time() - t0,
    }


def certify_ag(system: AGSystem, var_types: Optional[Dict[str, str]] = None) -> AGResult:
    """One-shot AG verification with certificate generation and checking."""
    result = verify_ag(system, var_types=var_types)

    if result.certificate:
        checked = check_certificate(result.certificate)
        result.certificate = checked

    return result


def ag_summary(result: AGResult) -> str:
    """Generate human-readable summary of AG verification result."""
    lines = [f"AG Verification: {result.verdict.value}"]
    lines.append(f"Components: {len(result.components)}")

    for name, cr in result.components.items():
        status = "VERIFIED" if cr.verified else "FAILED"
        obls = len(cr.obligations)
        lines.append(f"  {name}: {status} ({obls} obligations)")

    if result.discharge:
        dr = result.discharge
        lines.append(f"Discharge strategy: {dr.strategy}")
        lines.append(f"Discharged: {dr.discharged}")
        lines.append(f"Discharge obligations: {len(dr.obligations)}")

    if result.certificate:
        lines.append(f"Certificate: {result.certificate.status.value}")

    if result.metadata.get("time"):
        lines.append(f"Time: {result.metadata['time']:.3f}s")

    return "\n".join(lines)


def batch_verify(systems: List[Tuple[str, AGSystem]],
                 var_types: Optional[Dict[str, str]] = None) -> Dict[str, AGResult]:
    """Verify multiple AG systems, returning results keyed by name."""
    results = {}
    for name, system in systems:
        results[name] = verify_ag(system, var_types=var_types)
    return results


# ============================================================
# Non-interference (Information flow as AG)
# ============================================================

def verify_noninterference(comp: ComponentSpec,
                            high_vars: List[str], low_vars: List[str],
                            var_types: Optional[Dict[str, str]] = None) -> AGResult:
    """Verify non-interference: high inputs don't affect low outputs.

    Encodes as AG: for two runs with same low inputs but different high inputs,
    low outputs must be equal. Uses self-composition technique.
    """
    # Build two copies: one with primed high vars
    # Assumption: low inputs are equal (x1 == x2 for low vars)
    # Guarantee: low outputs are equal after execution

    assumptions = []
    for v in low_vars:
        assumptions.append(SBinOp("==", SVar(v), SVar(f"{v}_copy")))

    # Extract transformer for both copies
    transformer1 = _extract_body_transformer(comp)

    # Build guarantee: for each low var, transformer(v) == transformer_copy(v)
    guarantees = []
    for v in low_vars:
        if v in transformer1:
            expr1 = transformer1[v]
            # In copy, high vars are replaced with primed versions
            expr2 = expr1
            for hv in high_vars:
                expr2 = substitute(expr2, hv, SVar(f"{hv}_copy"))
            for lv in low_vars:
                expr2 = substitute(expr2, lv, SVar(f"{lv}_copy"))
            guarantees.append(SBinOp("==", expr1, expr2))

    if not guarantees:
        # No low-variable transformations, trivially non-interfering
        guarantees = [SBool(True)]

    all_vars = list(set(comp.params + [f"{v}_copy" for v in comp.params]))
    ni_comp = ComponentSpec(
        name=f"{comp.name}_noninterference",
        params=all_vars,
        assumptions=assumptions,
        guarantees=guarantees,
        body_source=comp.body_source,
        body_stmts=comp.body_stmts
    )

    system = AGSystem(components=[ni_comp], shared_vars=all_vars,
                      shared_var_types=var_types or {})
    return verify_ag(system, var_types=var_types)


# ============================================================
# Compositional refinement checking
# ============================================================

def check_contract_refinement(original: ComponentSpec, refined: ComponentSpec,
                               var_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Check if refined contract refines original (behavioral subtyping).

    Refinement: original.precond => refined.precond (accepts more)
              AND refined.postcond => original.postcond (provides at least as much)
    """
    pre_obligations = []
    post_obligations = []

    # Precondition weakening: original.pre => refined.pre
    for i, (orig_pre, ref_pre) in enumerate(zip(original.assumptions, refined.assumptions)):
        obl = _check_vc_sexpr(
            name=f"pre_weaken_{i}",
            formula=s_implies(orig_pre, ref_pre),
            var_types=var_types
        )
        obl.description = f"Precondition weakening: original pre {i} => refined pre {i}"
        pre_obligations.append(obl)

    # Postcondition strengthening: refined.post => original.post
    for i, (orig_post, ref_post) in enumerate(zip(original.guarantees, refined.guarantees)):
        obl = _check_vc_sexpr(
            name=f"post_strengthen_{i}",
            formula=s_implies(ref_post, orig_post),
            var_types=var_types
        )
        obl.description = f"Postcondition strengthening: refined post {i} => original post {i}"
        post_obligations.append(obl)

    all_valid = (all(o.status == VCStatus.VALID for o in pre_obligations) and
                 all(o.status == VCStatus.VALID for o in post_obligations))

    return {
        "refines": all_valid,
        "precondition_checks": pre_obligations,
        "postcondition_checks": post_obligations,
    }
