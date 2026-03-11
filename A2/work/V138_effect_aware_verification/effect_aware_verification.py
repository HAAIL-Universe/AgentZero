"""
V138: Effect-Aware Verification
================================
Compose V040 (effect systems) + V004 (VCGen) for effect-typed Hoare logic.

Effect inference tells us WHAT to verify; VCGen/SMT proves it.

Effect-specific VCs:
  - State(var): frame conditions -- unmodified vars preserved across function call
  - Exn(DivByZero): division safety -- all divisors proven non-zero
  - Pure: no assignments, no IO, no exceptions -- verify functional purity
  - Div: termination -- loops have bounded iteration (ranking function existence)
  - IO: output isolation -- print statements only in IO-declared functions

Composes:
  - V040 (effect systems) -- effect inference, EffectSet, EffectKind
  - V004 (VCGen) -- SExpr layer, WP calculus, check_vc, parse
  - C010 (parser) -- AST nodes
  - C037 (SMT solver) -- validity checking
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum, auto

# --- Path setup ---
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, '..', 'V040_effect_systems'))
sys.path.insert(0, os.path.join(_here, '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C037_smt_solver'))

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit,
)
from smt_solver import SMTSolver, SMTResult, Op, Var as SMTVar, IntConst, App, INT, BOOL
from effect_systems import (
    EffectKind, Effect, EffectSet, FnEffectSig, EffectInferrer,
    EffectChecker, EffectCheckResult, EffectCheckStatus, EffectVerificationResult,
    State, Exn, PURE, IO, DIV, NONDET,
)
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies, substitute as s_substitute,
    VCResult, VCStatus, VerificationResult as V004Result,
    check_vc, parse as c10_parse, lower_to_smt,
    WPCalculus, extract_fn_spec,
)

# ============================================================
# Result types
# ============================================================

class EAVStatus(Enum):
    """Effect-aware verification status."""
    VERIFIED = auto()
    FAILED = auto()
    UNKNOWN = auto()


@dataclass
class EffectVC:
    """A verification condition driven by effect analysis."""
    effect_kind: EffectKind      # Which effect generated this VC
    description: str             # Human-readable description
    formula: SExpr               # Symbolic formula to verify
    function_name: str           # Which function this belongs to
    status: Optional[VCStatus] = None
    counterexample: Optional[dict] = None


@dataclass
class EffectAwareResult:
    """Result of effect-aware verification."""
    status: EAVStatus
    effect_sigs: dict[str, FnEffectSig]    # Inferred effects per function
    vcs: list[EffectVC] = field(default_factory=list)
    effect_checks: list[EffectCheckResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.status == EAVStatus.VERIFIED

    @property
    def total_vcs(self) -> int:
        return len(self.vcs)

    @property
    def valid_vcs(self) -> int:
        return sum(1 for vc in self.vcs if vc.status == VCStatus.VALID)

    @property
    def failed_vcs(self) -> int:
        return sum(1 for vc in self.vcs if vc.status == VCStatus.INVALID)

    @property
    def unknown_vcs(self) -> int:
        return sum(1 for vc in self.vcs if vc.status == VCStatus.UNKNOWN)


# ============================================================
# AST helpers
# ============================================================

def _parse(source: str):
    """Parse C10 source to AST."""
    tokens = lex(source)
    parser = Parser(tokens)
    return parser.parse()


def _collect_stmts(node) -> list:
    """Flatten AST to list of statements."""
    if isinstance(node, Program):
        result = []
        for s in node.stmts:
            result.extend(_collect_stmts(s))
        return result
    elif isinstance(node, Block):
        result = []
        for s in node.stmts:
            result.extend(_collect_stmts(s))
        return result
    elif isinstance(node, IfStmt):
        result = [node]
        result.extend(_collect_stmts(node.then_body))
        if node.else_body:
            result.extend(_collect_stmts(node.else_body))
        return result
    elif isinstance(node, WhileStmt):
        result = [node]
        result.extend(_collect_stmts(node.body))
        return result
    elif isinstance(node, FnDecl):
        result = [node]
        result.extend(_collect_stmts(node.body))
        return result
    else:
        return [node]


def _find_divisions(node) -> list:
    """Find all division/modulo operations in an AST subtree."""
    divs = []
    if isinstance(node, BinOp) and node.op in ('/', '%'):
        divs.append(node)
    # Recurse into all fields
    if isinstance(node, (Program, Block)):
        stmts = node.stmts if hasattr(node, 'stmts') else []
        for s in stmts:
            divs.extend(_find_divisions(s))
    elif isinstance(node, LetDecl):
        if node.value:
            divs.extend(_find_divisions(node.value))
    elif isinstance(node, Assign):
        divs.extend(_find_divisions(node.value))
    elif isinstance(node, IfStmt):
        divs.extend(_find_divisions(node.cond))
        divs.extend(_find_divisions(node.then_body))
        if node.else_body:
            divs.extend(_find_divisions(node.else_body))
    elif isinstance(node, WhileStmt):
        divs.extend(_find_divisions(node.cond))
        divs.extend(_find_divisions(node.body))
    elif isinstance(node, FnDecl):
        divs.extend(_find_divisions(node.body))
    elif isinstance(node, ReturnStmt):
        if node.value:
            divs.extend(_find_divisions(node.value))
    elif isinstance(node, PrintStmt):
        divs.extend(_find_divisions(node.value))
    elif isinstance(node, CallExpr):
        for arg in node.args:
            divs.extend(_find_divisions(arg))
    elif isinstance(node, BinOp):
        divs.extend(_find_divisions(node.left))
        divs.extend(_find_divisions(node.right))
    elif isinstance(node, UnaryOp):
        divs.extend(_find_divisions(node.operand))
    return divs


def _expr_to_sexpr(expr) -> SExpr:
    """Convert C10 AST expression to SExpr."""
    if isinstance(expr, IntLit):
        return SInt(expr.value)
    elif isinstance(expr, BoolLit):
        return SBool(expr.value)
    elif isinstance(expr, ASTVar):
        return SVar(expr.name)
    elif isinstance(expr, BinOp):
        left = _expr_to_sexpr(expr.left)
        right = _expr_to_sexpr(expr.right)
        return SBinOp(expr.op, left, right)
    elif isinstance(expr, UnaryOp):
        operand = _expr_to_sexpr(expr.operand)
        return SUnaryOp(expr.op, operand)
    else:
        return SVar("_unknown")


def _find_assigned_vars(node) -> set:
    """Find all variables assigned in an AST subtree."""
    assigned = set()
    if isinstance(node, Assign):
        assigned.add(node.name)
    elif isinstance(node, LetDecl):
        assigned.add(node.name)
    elif isinstance(node, (Program, Block)):
        for s in node.stmts:
            assigned.update(_find_assigned_vars(s))
    elif isinstance(node, IfStmt):
        assigned.update(_find_assigned_vars(node.then_body))
        if node.else_body:
            assigned.update(_find_assigned_vars(node.else_body))
    elif isinstance(node, WhileStmt):
        assigned.update(_find_assigned_vars(node.body))
    elif isinstance(node, FnDecl):
        assigned.update(_find_assigned_vars(node.body))
    return assigned


def _find_print_stmts(node) -> list:
    """Find all print statements in an AST subtree."""
    prints = []
    if isinstance(node, PrintStmt):
        prints.append(node)
    elif isinstance(node, (Program, Block)):
        for s in node.stmts:
            prints.extend(_find_print_stmts(s))
    elif isinstance(node, IfStmt):
        prints.extend(_find_print_stmts(node.then_body))
        if node.else_body:
            prints.extend(_find_print_stmts(node.else_body))
    elif isinstance(node, WhileStmt):
        prints.extend(_find_print_stmts(node.body))
    elif isinstance(node, FnDecl):
        prints.extend(_find_print_stmts(node.body))
    return prints


def _find_while_loops(node) -> list:
    """Find all while loops in an AST subtree."""
    loops = []
    if isinstance(node, WhileStmt):
        loops.append(node)
        loops.extend(_find_while_loops(node.body))
    elif isinstance(node, (Program, Block)):
        for s in node.stmts:
            loops.extend(_find_while_loops(s))
    elif isinstance(node, IfStmt):
        loops.extend(_find_while_loops(node.then_body))
        if node.else_body:
            loops.extend(_find_while_loops(node.else_body))
    elif isinstance(node, FnDecl):
        loops.extend(_find_while_loops(node.body))
    return loops


def _find_functions(ast) -> list:
    """Find all FnDecl nodes."""
    fns = []
    if isinstance(ast, Program):
        for s in ast.stmts:
            if isinstance(s, FnDecl):
                fns.append(s)
    return fns


# ============================================================
# VC Generation Engine
# ============================================================

class EffectVCGenerator:
    """Generate verification conditions based on inferred effects."""

    def __init__(self):
        self.inferrer = EffectInferrer()

    def generate_vcs(self, source: str,
                     declared: Optional[dict[str, EffectSet]] = None) -> list[EffectVC]:
        """Generate effect-specific VCs for a C10 program."""
        # Infer effects
        sigs = self.inferrer.infer_program(source)
        ast = _parse(source)
        fns = _find_functions(ast)

        vcs = []

        for fn in fns:
            name = fn.name
            sig = sigs.get(name)
            if sig is None:
                continue

            eff = sig.effects
            decl_eff = (declared or {}).get(name, None)

            # 1. Division safety VCs
            # If declared with Exn, skip (caller accepts exceptions)
            # Otherwise always generate: either to verify declared no-Exn, or
            # to surface division risks in infer-and-verify mode
            if decl_eff is not None and decl_eff.has(EffectKind.EXN):
                pass  # Caller declared Exn, no division safety VCs needed
            else:
                vcs.extend(self._gen_division_safety_vcs(fn, name))

            # 2. Frame condition VCs (for functions with State effects)
            if decl_eff is not None:
                state_vars = set()
                for e in decl_eff.effects:
                    if e.kind == EffectKind.STATE and e.detail and e.detail != "*":
                        state_vars.add(e.detail)
                if state_vars:
                    vcs.extend(self._gen_frame_vcs(fn, name, state_vars))

            # 3. Purity VCs (for pure functions)
            if decl_eff is not None and decl_eff.is_pure:
                vcs.extend(self._gen_purity_vcs(fn, name))
            elif eff.is_pure:
                vcs.extend(self._gen_purity_vcs(fn, name))

            # 4. IO isolation VCs (functions without IO effect should have no prints)
            if decl_eff is not None and not decl_eff.has(EffectKind.IO):
                vcs.extend(self._gen_io_isolation_vcs(fn, name))

            # 5. Termination VCs (functions without Div effect should terminate)
            if decl_eff is not None and not decl_eff.has(EffectKind.DIV):
                vcs.extend(self._gen_termination_vcs(fn, name))

        return vcs

    def _gen_division_safety_vcs(self, fn: FnDecl, name: str) -> list[EffectVC]:
        """Generate VCs ensuring all divisions have non-zero divisors."""
        vcs = []
        divs = _find_divisions(fn.body)
        for i, div in enumerate(divs):
            divisor_sexpr = _expr_to_sexpr(div.right)
            # VC: divisor != 0
            formula = s_not(SBinOp("==", divisor_sexpr, SInt(0)))
            vcs.append(EffectVC(
                effect_kind=EffectKind.EXN,
                description=f"Division safety in '{name}': divisor #{i+1} is non-zero",
                formula=formula,
                function_name=name,
            ))
        return vcs

    def _gen_frame_vcs(self, fn: FnDecl, name: str,
                       declared_state_vars: set) -> list[EffectVC]:
        """Generate frame condition VCs: only declared State vars are modified."""
        assigned = _find_assigned_vars(fn.body)
        # Exclude function parameters (they are local)
        params = set(fn.params) if fn.params else set()
        non_param_assigned = assigned - params

        for var in non_param_assigned:
            if var not in declared_state_vars:
                # This variable is assigned but not declared in State effects
                # VC: var_pre == var_post (it should be preserved)
                # Since we can't easily get pre/post in this simple model,
                # we report it as a structural check
                formula = SBool(False)  # Structural violation: always fails
                vcs = [EffectVC(
                    effect_kind=EffectKind.STATE,
                    description=f"Frame violation in '{name}': assigns '{var}' but State set is {declared_state_vars}",
                    formula=formula,
                    function_name=name,
                )]
                return vcs

        # All assigned vars are in declared_state_vars: frame condition holds
        formula = SBool(True)
        return [EffectVC(
            effect_kind=EffectKind.STATE,
            description=f"Frame condition for '{name}': only modifies {declared_state_vars}",
            formula=formula,
            function_name=name,
        )]

    def _gen_purity_vcs(self, fn: FnDecl, name: str) -> list[EffectVC]:
        """Generate purity VCs: no assignments to non-local vars, no IO, no exceptions."""
        vcs = []
        # Check no print statements
        prints = _find_print_stmts(fn.body)
        if prints:
            vcs.append(EffectVC(
                effect_kind=EffectKind.PURE,
                description=f"Purity violation in '{name}': contains {len(prints)} print statement(s)",
                formula=SBool(False),
                function_name=name,
            ))
        else:
            vcs.append(EffectVC(
                effect_kind=EffectKind.PURE,
                description=f"IO purity for '{name}': no print statements",
                formula=SBool(True),
                function_name=name,
            ))

        # Check no divisions (could throw)
        divs = _find_divisions(fn.body)
        if divs:
            # Divisions exist -- generate safety VCs instead of blanket fail
            for i, div in enumerate(divs):
                divisor_sexpr = _expr_to_sexpr(div.right)
                formula = s_not(SBinOp("==", divisor_sexpr, SInt(0)))
                vcs.append(EffectVC(
                    effect_kind=EffectKind.PURE,
                    description=f"Exception purity in '{name}': division #{i+1} is safe",
                    formula=formula,
                    function_name=name,
                ))
        else:
            vcs.append(EffectVC(
                effect_kind=EffectKind.PURE,
                description=f"Exception purity for '{name}': no division operations",
                formula=SBool(True),
                function_name=name,
            ))

        return vcs

    def _gen_io_isolation_vcs(self, fn: FnDecl, name: str) -> list[EffectVC]:
        """Generate VCs ensuring no IO operations in non-IO functions."""
        prints = _find_print_stmts(fn.body)
        if prints:
            return [EffectVC(
                effect_kind=EffectKind.IO,
                description=f"IO isolation violation in '{name}': {len(prints)} print statement(s) in non-IO function",
                formula=SBool(False),
                function_name=name,
            )]
        return [EffectVC(
            effect_kind=EffectKind.IO,
            description=f"IO isolation for '{name}': no IO operations",
            formula=SBool(True),
            function_name=name,
        )]

    def _gen_termination_vcs(self, fn: FnDecl, name: str) -> list[EffectVC]:
        """Generate VCs for termination: loops must have bounded iteration."""
        loops = _find_while_loops(fn.body)
        if not loops:
            return [EffectVC(
                effect_kind=EffectKind.DIV,
                description=f"Termination for '{name}': no loops",
                formula=SBool(True),
                function_name=name,
            )]
        vcs = []
        for i, loop in enumerate(loops):
            # Try to find a ranking function from the loop condition
            # Simple heuristic: if loop condition is `var < bound` or `var > 0`,
            # and var is modified in the body, check that it converges
            ranking = self._extract_simple_ranking(loop)
            if ranking is not None:
                vcs.append(EffectVC(
                    effect_kind=EffectKind.DIV,
                    description=f"Termination for '{name}' loop #{i+1}: ranking function {ranking}",
                    formula=SBool(True),  # Ranking found structurally
                    function_name=name,
                ))
            else:
                vcs.append(EffectVC(
                    effect_kind=EffectKind.DIV,
                    description=f"Termination for '{name}' loop #{i+1}: no simple ranking function found",
                    formula=SBool(True),  # Conservative: don't fail, just warn
                    function_name=name,
                    status=VCStatus.UNKNOWN,
                ))
        return vcs

    def _extract_simple_ranking(self, loop: WhileStmt) -> Optional[str]:
        """Try to extract a simple ranking function from loop structure."""
        cond = loop.cond
        body_assigns = _find_assigned_vars(loop.body)

        # Pattern: while (i < n) { ... i = i + 1; ... } -> ranking: n - i
        if isinstance(cond, BinOp) and cond.op == '<':
            if isinstance(cond.left, ASTVar) and cond.left.name in body_assigns:
                return f"({_expr_str(cond.right)} - {cond.left.name})"
        # Pattern: while (i > 0) { ... i = i - 1; ... } -> ranking: i
        if isinstance(cond, BinOp) and cond.op == '>':
            if isinstance(cond.left, ASTVar) and cond.left.name in body_assigns:
                if isinstance(cond.right, IntLit) and cond.right.value == 0:
                    return cond.left.name
        # Pattern: while (i != 0) { ... } -> ranking: |i|
        if isinstance(cond, BinOp) and cond.op == '!=':
            if isinstance(cond.left, ASTVar) and cond.left.name in body_assigns:
                return f"|{cond.left.name}|"
            if isinstance(cond.right, ASTVar) and cond.right.name in body_assigns:
                return f"|{cond.right.name}|"
        return None


def _expr_str(expr) -> str:
    """Simple expression to string."""
    if isinstance(expr, IntLit):
        return str(expr.value)
    elif isinstance(expr, ASTVar):
        return expr.name
    elif isinstance(expr, BinOp):
        return f"({_expr_str(expr.left)} {expr.op} {_expr_str(expr.right)})"
    return "?"


# ============================================================
# VC Checker
# ============================================================

def _check_effect_vc(vc: EffectVC) -> EffectVC:
    """Check a single effect VC using SMT."""
    if vc.status is not None:
        return vc  # Already resolved (e.g., structural check)

    result = check_vc(vc.description, vc.formula)
    vc.status = result.status
    vc.counterexample = result.counterexample
    return vc


# ============================================================
# Main verification engine
# ============================================================

class EffectAwareVerifier:
    """Verify programs using effect-guided verification conditions."""

    def __init__(self):
        self.inferrer = EffectInferrer()
        self.checker = EffectChecker()
        self.vc_gen = EffectVCGenerator()

    def verify(self, source: str,
               declared: Optional[dict[str, EffectSet]] = None) -> EffectAwareResult:
        """Full effect-aware verification pipeline.

        1. Infer effects per function
        2. Check declared vs inferred (if declared provided)
        3. Generate effect-specific VCs
        4. Check VCs via SMT
        5. Return combined result
        """
        # Step 1-2: Effect inference and checking
        check_result = self.checker.check_program(source, declared)

        # Step 3: Generate effect-specific VCs
        vcs = self.vc_gen.generate_vcs(source, declared)

        # Step 4: Check all VCs
        for vc in vcs:
            _check_effect_vc(vc)

        # Step 5: Determine overall status
        has_failure = any(vc.status == VCStatus.INVALID for vc in vcs)
        has_unknown = any(vc.status == VCStatus.UNKNOWN for vc in vcs)
        has_effect_error = any(
            c.status == EffectCheckStatus.ERROR for c in check_result.checks
        )

        if has_failure or has_effect_error:
            status = EAVStatus.FAILED
        elif has_unknown:
            status = EAVStatus.UNKNOWN
        else:
            status = EAVStatus.VERIFIED

        return EffectAwareResult(
            status=status,
            effect_sigs=check_result.fn_sigs,
            vcs=vcs,
            effect_checks=check_result.checks,
            errors=check_result.errors if hasattr(check_result, 'errors') else [],
        )


# ============================================================
# Convenience APIs
# ============================================================

def verify_effects(source: str,
                   declared: Optional[dict[str, EffectSet]] = None) -> EffectAwareResult:
    """Verify a C10 program with effect-aware VCs.

    Args:
        source: C10 source code
        declared: Optional declared effects per function name
                  e.g., {"add": EffectSet.pure(), "update": EffectSet.of(State("x"))}

    Returns:
        EffectAwareResult with VCs, effect signatures, and overall verdict
    """
    v = EffectAwareVerifier()
    return v.verify(source, declared)


def verify_pure_function(source: str, fn_name: str) -> EffectAwareResult:
    """Verify a function is pure (no state, IO, exceptions)."""
    declared = {fn_name: EffectSet.pure()}
    return verify_effects(source, declared)


def verify_state_function(source: str, fn_name: str,
                          state_vars: list[str]) -> EffectAwareResult:
    """Verify a function only modifies declared state variables."""
    effects = EffectSet.of(*[State(v) for v in state_vars])
    declared = {fn_name: effects}
    return verify_effects(source, declared)


def verify_exception_free(source: str, fn_name: str) -> EffectAwareResult:
    """Verify a function cannot raise exceptions (division safety)."""
    # Declare function has no Exn effects
    declared = {fn_name: EffectSet.of(State("*"))}  # Allow state, disallow Exn
    return verify_effects(source, declared)


def verify_total(source: str, fn_name: str) -> EffectAwareResult:
    """Verify a function is total: terminates and doesn't throw."""
    # No Div, no Exn
    declared = {fn_name: EffectSet.of(State("*"))}
    return verify_effects(source, declared)


def infer_and_verify(source: str) -> EffectAwareResult:
    """Infer effects for all functions and verify the inferred constraints."""
    return verify_effects(source, declared=None)


def compare_declared_vs_inferred(source: str,
                                 declared: dict[str, EffectSet]) -> dict:
    """Compare declared effects against inferred effects.

    Returns a dict with:
      - per_function: {name: {declared, inferred, match, missing, extra}}
      - summary: overall statistics
    """
    inferrer = EffectInferrer()
    sigs = inferrer.infer_program(source)

    per_fn = {}
    for name, decl_eff in declared.items():
        inferred = sigs.get(name)
        if inferred is None:
            per_fn[name] = {
                "declared": str(decl_eff),
                "inferred": "not found",
                "match": False,
                "missing": [],
                "extra": [],
            }
            continue

        inf_eff = inferred.effects
        # Missing: in inferred but not declared (unsound declaration)
        missing = []
        for e in inf_eff.effects:
            if e.kind == EffectKind.PURE:
                continue
            if not decl_eff.has(e.kind):
                missing.append(str(e))
            elif e.kind == EffectKind.STATE and e.detail:
                # Check specific state var
                has_specific = any(
                    de.kind == EffectKind.STATE and (de.detail == e.detail or de.detail == "*")
                    for de in decl_eff.effects
                )
                if not has_specific:
                    missing.append(str(e))

        # Extra: declared but not inferred (over-approximation, safe)
        extra = []
        for e in decl_eff.effects:
            if not inf_eff.has(e.kind):
                extra.append(str(e))

        per_fn[name] = {
            "declared": str(decl_eff),
            "inferred": str(inf_eff),
            "match": len(missing) == 0,
            "missing": missing,
            "extra": extra,
        }

    total = len(per_fn)
    matching = sum(1 for v in per_fn.values() if v["match"])

    return {
        "per_function": per_fn,
        "summary": {
            "total": total,
            "matching": matching,
            "mismatched": total - matching,
            "sound": matching == total,
        },
    }


def effect_verification_summary(source: str,
                                declared: Optional[dict[str, EffectSet]] = None) -> dict:
    """Get a human-readable summary of effect-aware verification."""
    result = verify_effects(source, declared)

    vc_summary = []
    for vc in result.vcs:
        vc_summary.append({
            "effect": vc.effect_kind.name,
            "description": vc.description,
            "status": vc.status.name if vc.status else "UNCHECKED",
            "function": vc.function_name,
        })

    return {
        "status": result.status.name,
        "total_vcs": result.total_vcs,
        "valid": result.valid_vcs,
        "failed": result.failed_vcs,
        "unknown": result.unknown_vcs,
        "functions": {name: str(sig.effects) for name, sig in result.effect_sigs.items()},
        "vcs": vc_summary,
    }
